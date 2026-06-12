WORK_DIR=${WORK_DIR:-/projects/bhnl/xwu20/UMCTS-GRPO}
LOCAL_DIR=${LOCAL_DIR:-/projects/bhnl/xwu20/UMCTS-GRPO/data/singlehopqa}

mkdir -p "$LOCAL_DIR"

## process multiple dataset search format train file
DATA=nq
python "$WORK_DIR/scripts/data_process/qa_search_train_merge.py" --local_dir "$LOCAL_DIR" --data_sources "$DATA"

## process multiple dataset search format test file
DATA=nq,triviaqa,popqa
python "$WORK_DIR/scripts/data_process/qa_search_test_merge.py" --local_dir "$LOCAL_DIR" --data_sources "$DATA"
