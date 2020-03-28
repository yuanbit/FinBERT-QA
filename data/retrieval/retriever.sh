# Index documents using Lucene document indexer

#sh anserini/target/appassembler/bin/IndexCollection -collection JsonCollection \
 #-generator LuceneDocumentGenerator -threads 9 -input retrieval/collection_json \
 #-index retrieval/lucene-index-fiqa -storePositions -storeDocvectors -storeRawDocs

# Retrieve the top 500 candidate documents for the train set

anserini/target/appassembler/bin/SearchMsmarco  -hits 50 -threads 1 \
 -index retrieval/lucene-index-fiqa -qid_queries retrieval/train/train_questions.tsv \
 -output retrieval/train/cands_train_50.tsv

# Retrieve the top 500 candidate documents for the test set

anserini/target/appassembler/bin/SearchMsmarco  -hits 50 -threads 1 \
 -index retrieval/lucene-index-fiqa -qid_queries retrieval/test/test_questions.tsv \
 -output retrieval/test/cands_test_50.tsv

# Retrieve the top 500 candidate documents for the validation set

anserini/target/appassembler/bin/SearchMsmarco  -hits 50 -threads 1 \
 -index retrieval/lucene-index-fiqa -qid_queries retrieval/valid/valid_questions.tsv \
 -output retrieval/valid/cands_valid_50.tsv
