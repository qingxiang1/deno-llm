import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";
import { ChatAlibabaTongyi } from "@langchain/community/chat_models/alibaba_tongyi";
import "dotenv/config";
import { LLMChainExtractor } from "langchain/retrievers/document_compressors/chain_extract";
import { ContextualCompressionRetriever } from "langchain/retrievers/contextual_compression";

// process.env.LANGCHAIN_VERBOSE = "true";

async function run() {
  const directory = "./db/kongyiji";
  const embeddings = new AlibabaTongyiEmbeddings();
  const vectorstore = await FaissStore.load(directory, embeddings);

  const model = new ChatAlibabaTongyi({
    model: "qwen-turbo", // Available models: qwen-turbo, qwen-plus, qwen-max
    temperature: 1,
  });
  const compressor = LLMChainExtractor.fromLLM(model);

  const retriever = new ContextualCompressionRetriever({
    baseCompressor: compressor,
    baseRetriever: vectorstore.asRetriever(2),
  });
  const res = await retriever.invoke("茴香豆是做什么用的");
  console.log(res);
}

run();
