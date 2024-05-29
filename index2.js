import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import "dotenv/config";

const directory = "./db/kongyiji";
const embeddings = new AlibabaTongyiEmbeddings({});
const vectorStore = await FaissStore.load(directory, embeddings);

const retriever = vectorStore.asRetriever(2);
const res = await retriever.invoke("茴香豆是做什么用的");

console.log(res);

