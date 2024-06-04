import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";
import { ChatAlibabaTongyi } from "@langchain/community/chat_models/alibaba_tongyi";
import "dotenv/config";
import path from "path";
import { JSONChatHistory } from "../../JSONChatHistory/index";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import {
  RunnableSequence,
  RunnablePassthrough,
  RunnableWithMessageHistory,
  Runnable,
} from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { Document } from "@langchain/core/documents";

/**
 * 根据重写后的独立问题去读取数据库的中相关文档
 */
async function loadVectorStore() {
  const directory = path.join(__dirname, "../../db/qiu");
  const embeddings = new AlibabaTongyiEmbeddings();
  const vectorStore = await FaissStore.load(directory, embeddings);

  return vectorStore;
}

async function getRephraseChain() {
  const rephraseChainPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "给定以下对话和一个后续问题，请将后续问题重述为一个独立的问题。请注意，重述的问题应该包含足够的信息，使得没有看过对话历史的人也能理解。",
    ],
    new MessagesPlaceholder("history"),
    ["human", "将以下问题重述为一个独立的问题：\n{question}"],
  ]);

  const rephraseChain = RunnableSequence.from([
    rephraseChainPrompt,
    new ChatAlibabaTongyi({
      model: "qwen-turbo",
      temperature: 0.4,
    }),
    new StringOutputParser(),
  ]);

  return rephraseChain;
}

async function testRephraseChain() {
  const historyMessages = [
    new HumanMessage("你好，我是叮当猫"),
    new AIMessage("你好叮当猫"),
  ];
  const rephraseChain = await getRephraseChain();

  const question = "你觉得我的名字怎么样？";
  const standaloneQuestion = await rephraseChain.invoke({
    history: historyMessages,
    question,
  });

  console.log(standaloneQuestion);
}

export async function getRagChain(): Promise<Runnable> {
  const vectorStore = await loadVectorStore();
  const retriever = vectorStore.asRetriever(2);

  /**
   * 使用 retriever 获取相关文档，然后转换成纯字符串。
   * @param documents
   * @returns
   */
  const convertDocsToString = (documents: Document[]): string => {
    return documents.map((document) => document.pageContent).join("\n");
  };
  const contextRetrieverChain = RunnableSequence.from([
    (input) => input.standalone_question,
    retriever,
    convertDocsToString,
  ]);

  const SYSTEM_TEMPLATE = `
    你是一个熟读刘慈欣的《球状闪电》的终极原着党，精通根据作品原文详细解释和回答问题，你在回答时会引用作品原文。
    并且回答时仅根据原文，尽可能回答用户问题，如果原文中没有相关内容，你可以回答“原文中没有相关内容”，

    以下是原文中跟用户回答相关的内容：
    {context}
  `;

  /**
   * 包含历史记录信息的 prompt
   */
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    ["human", "现在，你需要基于原文，回答以下问题：\n{standalone_question}`"],
  ]);

  const model = new ChatAlibabaTongyi({
    model: "qwen-turbo",
  });
  const rephraseChain = await getRephraseChain();

  /**
   * 改写提问 => 根据改写后的提问获取文档 => 生成回复 的 rag chain
   */
  const ragChain = RunnableSequence.from([
    RunnablePassthrough.assign({
      standalone_question: rephraseChain,
    }),
    RunnablePassthrough.assign({
      context: contextRetrieverChain,
    }),
    prompt,
    model,
    new StringOutputParser(),
  ]);

  const chatHistoryDir = path.join(__dirname, "../../chat_data");

  /**
   * 使用 RunnableWithMessageHistory 去管理 history，给 chain 增加聊天记录的功能
   * 传给 getMessageHistory 的函数，需要根据用户传入的 sessionId 去获取初始的 chat history
   */
  const ragChainWithHistory = new RunnableWithMessageHistory({
    runnable: ragChain,
    getMessageHistory: (sessionId) =>
      new JSONChatHistory({ sessionId, dir: chatHistoryDir }),
    historyMessagesKey: "history",
    inputMessagesKey: "question",
  });

  return ragChainWithHistory;
}

async function run() {
  const ragChain = await getRagChain();

  const res = await ragChain.invoke(
    {
      // question: "什么是球状闪电？",
      question: "在国内大气物理学界，有人亲眼见过吗，在什么时候",
    },
    {
      configurable: { sessionId: "test-history" },
    }
  );

  console.log(res);
}

run();

