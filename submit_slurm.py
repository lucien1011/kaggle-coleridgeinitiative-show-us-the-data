def submit(cfg_path):
    import os,pickle
    
    from utils.objdict import ObjDict
    from utils.mkdir_p import mkdir_p
    from SLURMWorker.SLURMWorker import SLURMWorker

    cfg = ObjDict.read_from_file_python3(cfg_path)
    mkdir_p(cfg.slurm_job_dir)
    script_file_name = os.path.join(cfg.slurm_job_dir,cfg.slurm_cfg_name)
    
    worker = SLURMWorker()
    worker.make_sbatch_script(
            script_file_name,
            cfg.name,
            "kin.ho.lo@cern.ch",
            "1",
            "16gb",
            "72:00:00",
            cfg.slurm_job_dir,
            cfg.slurm_commands,
            gpu="geforce:1",
            )
    worker.sbatch_submit(script_file_name)

if __name__ == "__main__":
    import sys

    submit(sys.argv[1])
