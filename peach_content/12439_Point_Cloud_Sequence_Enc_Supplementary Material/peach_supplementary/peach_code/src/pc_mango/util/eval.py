import os


def get_best_checkpoint(checkpoint_path):
    def extract_metric_from_filename(filename):
        # Example for "best-checkpoint-02-0.01234567.ckpt"
        parts = filename.split('-')
        metric = parts[-1].split("=")[1]
        return float(metric.replace('.ckpt', ''))

    files = os.listdir(checkpoint_path)
    files = [f for f in files if not f == "last.ckpt"]
    extract_metric_from_filename(files[0])
    best_checkpoint = min(files, key=extract_metric_from_filename)
    return best_checkpoint