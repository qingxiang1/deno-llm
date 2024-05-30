import { FaissStore } from "@langchain/community/vectorstores/faiss";
// import { OpenAIEmbeddings } from "@langchain/openai";
import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";
import "dotenv/config";
import { ScoreThresholdRetriever } from "langchain/retrievers/score_threshold";

process.env.LANGCHAIN_VERBOSE = "true";

async function run() {
  const directory = "./db/kongyiji";
  const embeddings = new AlibabaTongyiEmbeddings();
  const vectorstore = await FaissStore.load(directory, embeddings);

  const retriever = ScoreThresholdRetriever.fromVectorStore(vectorstore, {
    minSimilarityScore: 0.8, // 定义了最小的相似度阈值，即文档向量和 query 向量相似度达到多少，就认为是可以被返回的。这个要根据文档类型设置，一般是 0.8 左右，可以避免返回大量的文档导致消耗过多的 token
    maxK: 3, // 一次最多返回多少条数据，主要是为了避免返回太多的文档造成 token 过度的消耗
    kIncrement: 1, // 定义了算法的布厂，你可以理解成 for 循环中的 i+k 中的 k。其逻辑是每次多获取 kIncrement 个文档，然后看这 kIncrement 个文档的相似度是否满足要求，满足则返回
  });
  const res = await retriever.invoke("茴香豆是做什么用的");
  console.log(res);
}

run();
