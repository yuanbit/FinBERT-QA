### Generate Training, Validation, Test Samples

```
python3 src/generate_dataset.py  [--train_cands TRAIN_CANDS] \
[--valid_cands VALID_CANDS] [--test_cands TEST_CANDS] \
[--train_label TRAIN_LABEL] [--valid_label VALID_LABEL] \
[--test_label TEST_LABEL] [--output_dir OUTPUT_DIR] \
[--cands_size CANDS_SIZE] [--neg_ans_size NEG_ANS_SIZE]

Arguments:
  TRAIN_CANDS - Path to the training candidates data in .tsv format. Each line should have three items: (questiod id, answer id, rank) separated by tab.
  VALID_CANDS - Path to the validation candidates data in .tsv format. Each line should have three items: (questiod id, answer id, rank) separated by tab.
  TEST_CANDS - Path to the testing candidates data in .tsv format. Each line should have three items: (questiod id, answer id, rank) separated by tab.
  TRAIN_LABEL - Path to the training label data in .pickle format.
  VALID_LABEL - Path to the validation label data in .pickle format.
  TEST_LABEL - Path to the testing label data in .pickle format.
  OUTPUT_DIR - The output directory where the generated data will be stored.
  CANDS_SIZE - Number of candidates per question.
```
#### Example
```
python3 src/generate_dataset.py --train_cands data/train/cands_train_500.tsv \
--valid_cands data/valid/cands_valid_500.tsv --test_cands data/test/cands_test_500.tsv \
--train_label data/train/qid_rel_train.pickle --valid_label data/valid/qid_rel_valid.pickle \
--test_label data/test/qid_rel_test.pickle --output_dir data/processed \
--cands_size 50
```
