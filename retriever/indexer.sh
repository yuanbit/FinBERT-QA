# Index documents using Lucene document indexer

sh anserini/target/appassembler/bin/IndexCollection -collection JsonCollection \
 -generator LuceneDocumentGenerator -threads 9 -input retrieval/collection_json \
 -index retrieval/lucene-index-fiqa -storePositions -storeDocvectors -storeRawDocs
