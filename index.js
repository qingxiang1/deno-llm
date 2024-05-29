import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import "dotenv/config";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";

const run = async () => {
  const loader = new TextLoader("./data/2.txt");
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 100,
    chunkOverlap: 20,
  });

  const splitDocs = await splitter.splitDocuments(docs);

  const embeddings = new AlibabaTongyiEmbeddings({});
  const vectorStore = await FaissStore.fromDocuments(splitDocs, embeddings);

  const directory = "./db/kongyiji";
  await vectorStore.save(directory);
};

run();
