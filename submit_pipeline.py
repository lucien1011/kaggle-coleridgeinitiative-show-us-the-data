import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_path')
    parser.add_argument('jobs')
    parser.add_argument('--slurm_cfg_name',default='slurm_cfg',)
    return parser.parse_args()

def submit(cfg_path,jobs,cfg_name='slurm_cfg',):
    import os,pickle
    
    from utils.objdict import ObjDict
    from utils.mkdir_p import mkdir_p
    from SLURMWorker.SLURMWorker import SLURMWorker

    cfg = ObjDict.read_from_file_python3(cfg_path,cfg_name)
    mkdir_p(cfg.slurm_job_dir)
    script_file_name = os.path.join(cfg.slurm_job_dir,cfg.slurm_cfg_name)
    
    worker = SLURMWorker()
    slurm_commands = """
cd {base_path}
source setup_hpg.sh
python3 {pyscript} {cfg_path} {mode}
""".format(
            pyscript="run.py",
            cfg_path="config/"+cfg.name+".py",
            mode=jobs,
            base_path=os.environ['BASE_PATH'],
            )
    worker.make_sbatch_script(
            script_file_name,
            cfg.name,
            cfg.email,
            "1",
            cfg.memory,
            cfg.time,
            cfg.slurm_job_dir,
            slurm_commands,
            gpu=cfg.gpu,
            )
    worker.sbatch_submit(script_file_name)

if __name__ == "__main__":
    
    args = parse_arguments()
    submit(args.cfg_path,args.jobs,args.slurm_cfg_name)
