# Index documents using Lucene document indexer

sh anserini/target/appassembler/bin/IndexCollection -collection JsonCollection \
 -generator LuceneDocumentGenerator -threads 9 -input collection_json \
 -index lucene-index-fiqa -storePositions -storeDocvectors -storeRawDocs
