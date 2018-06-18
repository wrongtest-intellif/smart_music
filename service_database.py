import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions, DirectoryReader
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser


def build_index(document_path, dir_path):
    lucene.initVM()  
    index_dir = SimpleFSDirectory(Paths.get(dir_path))
    analyzer = StandardAnalyzer()   
    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    index_writer = IndexWriter(index_dir, config)

    t1 = FieldType()
    t1.setStored(True)
    t1.setTokenized(True)
    t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS)
    
    t2 = FieldType()
    t2.setStored(True)
    t2.setTokenized(False)
    
    with open(document_path) as input_file:
        for line in input_file:
            segs = line.strip().split(" ")
            music_path, music_tags = segs[0], segs[1].split(",")

            document = Document()
            document.add(Field("content", " ".join(music_tags), t1))
            document.add(Field("url", music_path, t2))
            index_writer.addDocument(document)  

    index_writer.close()


def search(music_tags, dir_path):
    lucene.initVM()
      
    query_str ="content:" + " ".join(music_tags)
    index_dir = SimpleFSDirectory(Paths.get(dir_path))
    lucene_analyzer= StandardAnalyzer()
    lucene_searcher= IndexSearcher(DirectoryReader.open(index_dir))
          
    my_query = QueryParser("content", lucene_analyzer).parse(query_str)
    total_hits = lucene_searcher.search(my_query, 50)
            
    for hit in total_hits.scoreDocs:
        doc = lucene_searcher.doc(hit.doc)
        print doc


if __name__ == "__main__":
    build_index("sample2.txt", "lucene_index")
    search(["happy"], "lucene_index")
                                                
                                                  
                                                   
