import sys
import os
import tempfile
import time

sys.path.insert(1, os.path.join("..", "..", ".."))
import h2o
from tests import pyunit_utils
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.gbm import H2OGradientBoostingEstimator


def import_iris():
    return h2o.import_file(path=pyunit_utils.locate("smalldata/iris/iris_wheader.csv"))


def gbm_start(grid_id, export_dir, train, hyper_parameters):
    grid = H2OGridSearch(
        H2OGradientBoostingEstimator,
        grid_id=grid_id,
        hyper_params=hyper_parameters,
        export_checkpoints_dir=export_dir,
        checkpoint_frames=True
    )
    grid.start(x=list(range(4)), y=4, training_frame=train)
    return grid


def gbm_resume(grid, train):
    grid.train(x=list(range(4)), y=4, training_frame=train)


def grid_ft_resume(import_data, grid_id, hyper_parameters, start_grid, resume_grid):
    export_dir = tempfile.mkdtemp()
    print("Using directory %s" % export_dir)
    grid_size = 1
    for p in hyper_parameters:
        grid_size *= len(hyper_parameters[p])
    print("Grid size %d" % grid_size)
    train = import_data()
    print("Starting baseline grid")
    grid = start_grid(grid_id, export_dir, train, hyper_parameters)
    grid_in_progress = None
    times_waited = 0
    while (times_waited < 20) and (grid_in_progress is None or len(grid_in_progress.model_ids) == 0):
        time.sleep(5)  # give it tome to train some models
        times_waited += 1
        try:
            grid_in_progress = h2o.get_grid(grid_id)
        except IndexError:
            print("no models trained yet")
    grid.cancel()

    grid = h2o.get_grid(grid_id)
    old_grid_model_count = len(grid.model_ids)
    for x in sorted(grid.model_ids):
        print(x)
    print("Baseline grid has %d models" % old_grid_model_count)
    h2o.remove_all()

    loaded = h2o.load_grid("%s/%s" % (export_dir, grid_id), load_frames=True)
    assert loaded is not None
    assert len(grid.model_ids) == old_grid_model_count
    loaded_train = h2o.H2OFrame.get_frame(train.frame_id)
    assert loaded_train is not None, "Train frame was not loaded"
    loaded.hyper_params = hyper_parameters
    print("Starting final grid")
    resume_grid(loaded, train)
    for x in sorted(loaded.model_ids):
        print(x)
    print("Newly grained grid has %d models" % len(loaded.model_ids))
    assert len(loaded.model_ids) == grid_size, "The full grid was not trained."
    

def grid_ft_resume_test():
    grid_ft_resume(
        import_iris, "gbm_grid_ft", {
            "learn_rate": [0.01, 0.02, 0.03, 0.04],
            "ntrees": [100, 110, 120, 130]
        }, gbm_start, gbm_resume
    )


if __name__ == "__main__":
    pyunit_utils.standalone_test(grid_ft_resume_test)
else:
    grid_ft_resume_test()
