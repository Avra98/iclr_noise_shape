import json,sys
from pathlib import Path
from probe import parser,run
r=Path('/scratch/users/ghoshavr/eoss/results/APPENDIX_DIRECTIONAL_LADDERS_20260906')
c=json.loads((r/'manifest.json').read_text())['cases'][int(sys.argv[1])]
a=parser().parse_args(['--checkpoint',c['checkpoint'],'--metadata',c['metadata'],'--outdir',str(r/'runs'/c['id']),'--steps','4096','--replay-seed','920201','--frame-n','0','--frame-iters','40','--random-axes','0','--probe-every','256','--probe-draws','64','--probe-chunk','256','--train-chunk','256','--flush-every','256','--device','cuda','--threads','2'])
run(a)
