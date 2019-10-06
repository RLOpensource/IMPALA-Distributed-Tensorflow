ps_hosts="localhost:2222"
worker_hosts="localhost:2223,localhost:2224,localhost:2225,localhost:2226,localhost:2227,localhost:2228,localhost:2229,localhost:2230,localhost:2231,localhost:2232,localhost:2233,localhost:2234,localhost:2235,localhost:2236,localhost:2237,localhost:2238,localhost:2239,localhost:2240,localhost:2241,localhost:2242"

python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=ps --task_index=0 & # 22

python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=0 & #23
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=1 & #24
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=2 & #25
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=3 & #26
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=4 & #27
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=5 & #28
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=6 & #29
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=7 & #30
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=8 & #31
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=9 & #32
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=10 & #33
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=11 & #34
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=12 & #35
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=13 & #36
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=14 & #37
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=15 & #38
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=16 & #39
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=17 & #40
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=18 & #41
python trainer.py  --ps_hosts=$ps_hosts  --worker_hosts=$worker_hosts  --job_name=worker --task_index=19 & #42