import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";
import { ChatAlibabaTongyi } from "@langchain/community/chat_models/alibaba_tongyi";
import { MultiQueryRetriever } from "langchain/retrievers/multi_query";
import "faiss-node";
import "dotenv/config";

async function run() {
  const directory = "./db/kongyiji";
  const embeddings = new AlibabaTongyiEmbeddings({});
  const vectorstore = await FaissStore.load(directory, embeddings);

  const model = new ChatAlibabaTongyi({
    model: "qwen-turbo", // Available models: qwen-turbo, qwen-plus, qwen-max
    temperature: 1,
  });
  const retriever = MultiQueryRetriever.fromLLM({
    llm: model,
    retriever: vectorstore.asRetriever(3), // 每次会检索三条数据，对每个 query
    queryCount: 3, // 默认值是 3，也就意味着会对每条输入，都会用 llm 改写生成三条不同写法和措词，但表示同样意义的 query
    verbose: true, // 设置为 true 会打印出 chain 内部的详细执行过程方便 debug
  });
  const res = await retriever.invoke("茴香豆是做什么用的");

  console.log(res);
}

run();
