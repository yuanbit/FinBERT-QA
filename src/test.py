from utils import *

train_set = load_pickle('../fiqa/data/processed_data/train_set_50.pickle')
valid_set = load_pickle('../fiqa/data/processed_data/valid_set_50.pickle')
test_set = load_pickle('../fiqa/data/processed_data/test_set_50.pickle')

save_pickle('../fiqa/data/test_data/train_toy.pickle', train_set[:100])
save_pickle('../fiqa/data/test_data/valid_toy.pickle', train_set[:10])
save_pickle('../fiqa/data/test_data/test_toy.pickle', train_set[:10])
