#!/usr/bin/env python3
"""Fixed-coordinate occupancy and direction-resolved SGD tangent diagnostics.

All tangent derivatives use the exact pre-update state and training batch.
Fixed scalar compressions, reset-direction norm gains and propagated products
are stored separately; none is silently substituted for another.
"""
import argparse,copy,hashlib,json,os,signal,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from stability_common import hvp_chunked,build_arch,_flat_grad,grad_chunked,seed_everything
from fixed_frame_vision import leading_ritz,diagnostic_rng,provenance,regenerate_basis,atomic_json,atomic_npz,sha256
ROOT=Path('/scratch/users/ghoshavr/eoss/results/OVERLEAF_FIGURES_20260906')
STOP=False

def kernel(net,lf,x,y,directions,scalar=True):
    params=[p for p in net.parameters() if p.requires_grad]
    loss=lf(net(x),y)
    g=_flat_grad(loss,params,create_graph=True)
    hs=[]
    for i,q in enumerate(directions):
        hs.append(_flat_grad(torch.dot(g,q.detach()),params,retain_graph=(i+1<len(directions) or scalar)).detach())
    hq=torch.stack(hs)
    if scalar:
        hg=_flat_grad(torch.dot(g,g.detach()),params).detach()
        denom=g.detach().double().square().sum()
        s=float((g.detach().double()*hg.double()).sum()/denom) if float(denom)>0 else float('nan')
    else:s=float('nan')
    return g.detach(),hq,float(loss.detach()),s

def flat(net):return torch.cat([p.detach().reshape(-1) for p in net.parameters() if p.requires_grad])

def tangent_step(q,hq,eta):
    v=q.double()-eta*hq.double()
    n0=q.double().norm(dim=1);n1=v.norm(dim=1)
    logs=torch.log(n1/n0)
    if not bool(torch.isfinite(logs).all()):raise FloatingPointError('Nonfinite or annihilated tangent; explicit termination')
    return (v/n1[:,None]).to(q.dtype),logs

def validate(out):
    torch.manual_seed(137)
    net=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.SiLU(),torch.nn.Linear(3,2)).double()
    x=torch.randn(5,2,dtype=torch.float64);y=torch.randn(5,2,dtype=torch.float64)
    lf=lambda a,b:(a-b).square().sum(1).mean();w=flat(net)
    q=torch.randn(3,len(w),dtype=w.dtype);q/=q.norm(dim=1,keepdim=True)
    pp=dict(net.named_parameters());keys=list(pp);shapes=[z.shape for z in pp.values()];sizes=[z.numel() for z in pp.values()]
    def f(z):return lf(torch.func.functional_call(net,{k:v.view(s) for k,v,s in zip(keys,z.split(sizes),shapes)},(x,)),y)
    H=torch.func.hessian(f)(w);g0=torch.func.grad(f)(w)
    g,h,l,s=kernel(net,lf,x,y,q)
    eta=.08;qn,logs=tangent_step(q,h,eta);v=q-eta*q@H.T
    e=dict(gradient=float((g-g0).abs().max()),hvp=float((h-q@H.T).abs().max()),scalar=abs(s-float(g0@H@g0/(g0@g0))),log_gain=float((logs-v.norm(dim=1).log()).abs().max()))
    eps=1e-5;fd=((w+eps*q[0]-eta*torch.func.grad(f)(w+eps*q[0]))-(w-eps*q[0]-eta*torch.func.grad(f)(w-eps*q[0])))/(2*eps)
    e['same_batch_map_finite_difference']=float((fd-v[0]).abs().max())
    assert max(e.values())<1e-7,e
    atomic_json(Path(out),dict(passed=True,errors=e,loss='mean examples, sum outputs'))

def run(c,steps_override=None):
    global STOP
    out=ROOT/'branches'/c['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'meta.json').exists():raise RuntimeError('Output already exists; choose new branch id')
    torch.set_num_threads(2);seed_everything(c['stream']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    dev=torch.device('cuda');cp_path=Path(c['checkpoint']);cp=torch.load(cp_path,map_location='cpu',weights_only=False)
    cfg,source,mp=provenance(cp_path,cp,Path(c['metadata']))
    x,y,net,lf,opt=build_arch(cfg['arch'],dev,eta=c['eta'],n_train=cfg['n_train'],dataset_seed=cfg['dataset_seed'],init_seed=cfg['init_seed'],init_scale=cfg['init_scale'])
    net.load_state_dict(cp['model_state_dict']);del cp
    for group in opt.param_groups:group.update(lr=c['eta'],momentum=0,weight_decay=0)
    net.train();w0=flat(net).double();N=len(w0)
    fr=torch.load(c['frame'],map_location='cpu',weights_only=False)
    assert fr['checkpoint_sha256']==sha256(cp_path)
    top_random=regenerate_basis(fr['top_ritz_vector'],fr['random_axis_seed'],2).to(dev)
    g0=grad_chunked(net,lf,x,y,128);g0/=g0.norm()
    fixed=torch.stack([top_random[0],g0,top_random[1],top_random[2]])
    fixed=(fixed.double()/fixed.double().norm(dim=1,keepdim=True)).float();fixed64=fixed.double();norms2=fixed64.square().sum(1)
    q=fixed[[0,1,2]].clone()
    axis_names=['fixed_top_ritz','fixed_gradient','fixed_random_1','fixed_random_2'];moving_names=['top_initialized','gradient_initialized','random_initialized']
    steps=steps_override or c.get('steps',4096);eta=c['eta'];batch=cfg['batch'];params=[p for p in net.parameters() if p.requires_grad]
    # Every randomness source has a separate generator; probes cannot alter the trajectory.
    train_gen=torch.Generator(device=dev).manual_seed(c['stream']);probe_gen=torch.Generator(device=dev).manual_seed(81301)
    frozen_gen=torch.Generator(device=dev).manual_seed(73111)
    trace={k:[] for k in ['step','coordinates','batch_loss','scalar_s','fixed_rayleigh','fixed_norm_log_gain','fixed_orthogonal_gain_sq','moving_log_gain','moving_alignment_top','gradient_alignment','parameter_displacement']}
    fresh={k:[] for k in ['probe_step','fixed_gain','fixed_norm_log_gain','scalar_gain','pair_group','draw_index']}
    alignment=[];precision=[];full_losses=[];started=time.time();done=0
    meta=dict(c,architecture=cfg['arch'],eta=eta,batch=batch,requested_steps=steps,axis_names=axis_names,moving_names=moving_names,checkpoint_sha256=sha256(cp_path),parent_step=cfg['checkpoint_step'],source_metadata=mp,
        fixed_frame_eigenpair=fr['eigenpair'],fixed_frame_examples=fr['hessian_examples'],fixed_gradient_examples=len(x),axis_gram=(fixed64@fixed64.T).cpu().tolist(),
        training_precision='float32 with TF32 disabled',reduction_precision='float64 for projections, Rayleigh quotients and norm ratios',
        sampling=c['sampling'],optimizer='plain SGD; no clipping, momentum or weight decay',loss='mean examples, sum outputs squared error',
        coordinates='q^T(theta_t-theta_checkpoint); fixed axes and center throughout; pre-update state',
        moving='continuous product of every training-update Jacobian, normalized between updates, three initial directions; no direction resetting',
        fixed='scalar compression 1-eta q^T H_B q / ||q||^2; full reset-axis norm ||(I-eta H_B)q||/||q|| also saved',
        precision_scope='sparse same-state same-vector float64 derivative comparisons; no global asymptotic error bound',
        fresh_probe_pairing='Same independently generated minibatches within each adjacent512k/512k+1 state pair, indexed by pair_group/draw_index',branch_is_exact_resume=False,source_sha256=sha256(Path(__file__)),gpu=torch.cuda.get_device_name(),status='running',completed_updates=0)
    atomic_json(out/'meta.json',meta)
    def flush():
        atomic_npz(out/'trace.npz',{**{k:np.asarray(v) for k,v in trace.items()},'eta':np.asarray(eta),'axis_names':np.asarray(axis_names),'moving_names':np.asarray(moving_names)})
        atomic_npz(out/'fresh_probes.npz',{**{k:np.asarray(v) for k,v in fresh.items()},'eta':np.asarray(eta)})
        atomic_json(out/'precision.json',precision)
        atomic_json(out/'full_loss.json',full_losses)
        atomic_json(out/'eigenvector_alignment.json',alignment)
        meta.update(completed_updates=done,elapsed_seconds=time.time()-started);atomic_json(out/'meta.json',meta)
    def sample_probe(t,draws):
        # Common fresh batches across an adjacent state pair, isolated from training.
        pair_group=t//512;probe_gen.manual_seed(81301+1009*pair_group)
        for j in range(draws):
            ix=torch.randperm(len(x),device=dev,generator=probe_gen)[:batch]
            _,hq,_,s=kernel(net,lf,x[ix],y[ix],fixed)
            h=(fixed64*hq.double()).sum(1)/norms2
            fresh['probe_step'].append(t);fresh['pair_group'].append(pair_group);fresh['draw_index'].append(j);fresh['fixed_gain'].append((1-eta*h).cpu().numpy());fresh['scalar_gain'].append(1-eta*s)
            fresh['fixed_norm_log_gain'].append(torch.log((fixed64-eta*hq.double()).norm(dim=1)/norms2.sqrt()).cpu().numpy())
    def frozen(tag,horizon):
        z=fixed[[0,1,2]].clone();logs=[]
        for j in range(horizon):
            ix=torch.randperm(len(x),device=dev,generator=frozen_gen)[:batch]
            _,hq,_,_=kernel(net,lf,x[ix],y[ix],z,scalar=False)
            z,lg=tangent_step(z,hq,eta);logs.append(lg.cpu().numpy())
        atomic_npz(out/f'frozen_{tag}.npz',dict(log_gain=np.asarray(logs),eta=np.asarray(eta),moving_names=np.asarray(moving_names),burn_in=np.asarray(min(64,horizon//4))))
    def eigen_probe(t):
        with diagnostic_rng(dev):
            v, info = leading_ritz(lambda z: hvp_chunked(net,lf,x,y,z,256),N,dev,19817,40,1e-3)
        alignment.append(dict(step=t,alignment=abs(float(v.double()@fixed64[0]))/float(v.double().norm()*fixed64[0].norm()),**info))
        print(json.dumps(dict(eigen_probe=alignment[-1])),flush=True)
    def loss_all(t):
        if t not in [a['step'] for a in alignment]:eigen_probe(t)
        with torch.no_grad():lv=sum(float(lf(net(x[i:i+256]),y[i:i+256]))*len(x[i:i+256])/len(x) for i in range(0,len(x),256))
        full_losses.append(dict(step=t,loss=lv))
    def precision_check(t,ix,directions,h32,s32):
        model64=copy.deepcopy(net).double()
        gd,h64,_,sd=kernel(model64,lf,x[ix].double(),y[ix].double(),directions.double())
        qd=directions[4:].double()
        _,l64=tangent_step(qd,h64[4:],eta);_,l32=tangent_step(directions[4:],h32[4:],eta)
        precision.append(dict(step=t,scalar_s_fp32=s32,scalar_s_fp64=sd,moving_log_fp32=l32.cpu().tolist(),moving_log_fp64=l64.cpu().tolist(),moving_log_abs_difference=(l32-l64).abs().cpu().tolist(),hvp_relative_error=float((h32.double()-h64).norm()/h64.norm())))
        del model64
    signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__('STOP',True));signal.signal(signal.SIGINT,lambda *_:globals().__setitem__('STOP',True))
    loss_all(0);sample_probe(0,c.get('initial_draws',128));frozen('initial',c.get('frozen_horizon',256));flush()
    last_epoch=None;permutation=None
    for t in range(steps):
        if c['sampling']=='iid_subsets':ix=torch.randperm(len(x),device=dev,generator=train_gen)[:batch]
        else:
            epoch,pos=divmod(cfg['checkpoint_step']+t,len(x)//batch)
            if epoch!=last_epoch:
                train_gen.manual_seed(c['stream']+epoch);permutation=torch.randperm(len(x),device=dev,generator=train_gen);last_epoch=epoch
            ix=permutation[pos*batch:(pos+1)*batch]
        directions=torch.cat([fixed,q]);g,hq,lv,s=kernel(net,lf,x[ix],y[ix],directions)
        if not np.isfinite(lv) or lv>1e8 or not bool(torch.isfinite(g).all()):meta['status']='escaped_or_nonfinite';break
        if t in c.get('precision_steps',[0,512,2048]):precision_check(t,ix,directions,hq,s)
        delta=flat(net).double()-w0;hf=hq[:4].double();h=(fixed64*hf).sum(1)/norms2
        newq,mlog=tangent_step(q,hq[4:],eta)
        values=dict(step=t,coordinates=(fixed64@delta).cpu().numpy(),batch_loss=lv,scalar_s=s,fixed_rayleigh=h.cpu().numpy(),fixed_norm_log_gain=torch.log((fixed64-eta*hf).norm(dim=1)/norms2.sqrt()).cpu().numpy(),fixed_orthogonal_gain_sq=(eta**2*(hf.square().sum(1)/norms2-h.square())).cpu().numpy(),moving_log_gain=mlog.cpu().numpy(),moving_alignment_top=(q.double()@fixed64[0]).cpu().numpy(),gradient_alignment=((fixed64@g.double())/(norms2.sqrt()*g.double().norm())).cpu().numpy(),parameter_displacement=float(delta.norm()))
        for k,v in values.items():trace[k].append(v)
        opt.zero_grad()
        offset=0
        for p in params:p.grad=g[offset:offset+p.numel()].view_as(p);offset+=p.numel()
        opt.step();q=newq;done=t+1
        if float(delta.norm())>1e6:meta['status']='displacement_cutoff';break
        # Both parities sampled explicitly at every checkpoint pair.
        if done%512 in [0,1] or done==steps:sample_probe(done,c.get('probe_draws',32))
        if done in [2049,2305,2561]:eigen_probe(done)
        if done%256==0:
            loss_all(done);flush();print(json.dumps(dict(branch=c['id'],step=done,loss=lv,elapsed=time.time()-started)),flush=True)
        if STOP:meta['status']='interrupted';break
    else:meta['status']='completed'
    if meta['status']=='completed':frozen('final',c.get('frozen_horizon',256))
    loss_all(done);flush();print(json.dumps(meta),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=ROOT/'manifest.json');p.add_argument('--index',type=int);p.add_argument('--validate',type=Path);p.add_argument('--steps',type=int)
    a=p.parse_args()
    if a.validate:validate(a.validate)
    else:run(json.loads(a.manifest.read_text())['branches'][a.index],a.steps)
if __name__=='__main__':main()
