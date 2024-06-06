import { readFileSync } from "fs";
import path from "path";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
// import { ChatOpenAI } from "@langchain/openai";
import { ChatAlibabaTongyi } from "@langchain/community/chat_models/alibaba_tongyi";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { z } from "zod";
import "dotenv/config";
import { DynamicStructuredTool } from "@langchain/core/tools";
import {
  AgentExecutor,
  createOpenAIToolsAgent,
  // createReactAgent,
} from "langchain/agents";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import readline from "readline";
// import { Calculator } from "@langchain/community/tools/calculator";

process.env.LANGCHAIN_TRACING_V2 = "true";

const mbtiInfoBuffer = readFileSync(path.join(__dirname, "./type-info.json"));
const mbtiInfo = JSON.parse(mbtiInfoBuffer.toString());
const mbtiList = [
  "ISTJ",
  "ISFJ",
  "INFJ",
  "INTJ",
  "ISTP",
  "ISFP",
  "INFP",
  "INTP",
  "ESTP",
  "ESFP",
  "ENFP",
  "ENTP",
  "ESTJ",
  "ESFJ",
  "ENFJ",
  "ENTJ",
] as [string, ...string[]];

async function getMBTIChatChain() {
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "你是一个共情能力非常强的心理医生，并且很了解MBTI（迈尔斯-布里格斯性格类型指标)的各种人格类型，你的任务是根据来访者的 MBTI 和问题，给出针对性的情感支持，你的回答要富有感情、有深度和充足的情感支持，引导来访者乐观积极面对问题",
    ],
    [
      "human",
      "用户的 MBTI 类型是{type}, 这个类型的特点是{info}, 他的问题是{question}",
    ],
  ]);

  const model = new ChatAlibabaTongyi({
    model: "qwen-max",
  });
  
  const mbtiChain = RunnableSequence.from([
    prompt,
    model,
    new StringOutputParser(),
  ]);

  return mbtiChain;
}

async function getAgent() {
  const mbtiChatChain = await getMBTIChatChain();

  const mbtiTool = new DynamicStructuredTool({
    name: "get-mbti-chat",
    schema: z.object({
      type: z.enum(mbtiList).describe("用户的 MBTI 类型"),
      question: z.string().describe("用户的问题"),
    }),
    func: async ({ type, question }) => {
      const info = mbtiInfo[type];

      const res = await mbtiChatChain.invoke({ type, question, info });
      return res;
    },
    description: "根据用户的问题和 MBTI 类型，回答用户的问题",
  });

  const tools = [mbtiTool];

  const agentPrompt = await ChatPromptTemplate.fromMessages([
    [
      "system",
      "你是一个用户接待的 agent，通过自然语言询问用户的 MBTI 类型和问题，直到你有足够的信息调用 get-mbti-chat 来回答用户的问题",
    ],
    new MessagesPlaceholder("history_message"),
    ["human", "{input}"],
    new MessagesPlaceholder("agent_scratchpad"),
  ]);

  const llm = new ChatAlibabaTongyi({
    model: "qwen-max",
    temperature: 0.4,
  });

  const agent = await createOpenAIToolsAgent({
    llm,
    tools,
    prompt: agentPrompt,
  });

  const agentExecutor = new AgentExecutor({
    agent,
    tools,
  });

  const messageHistory = new ChatMessageHistory();

  const agentWithChatHistory = new RunnableWithMessageHistory({
    runnable: agentExecutor,
    getMessageHistory: () => messageHistory,
    inputMessagesKey: "input",
    historyMessagesKey: "history_message",
  });

  return agentWithChatHistory;
}

async function main() {
  const agent = await getAgent();

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  function chat() {
    rl.question("User: ", async (input) => {
      if (input.toLowerCase() === "exit") {
        rl.close();
        return;
      }

      const response = await agent.invoke(
        {
          input,
        },
        {
          configurable: {
            sessionId: "no-used",
          },
        }
      );

      console.log("Agent: ", response.output);

      chat();
    });
  }

  console.log("请输入问题，输入 exit 退出聊天。");
  chat();
}

main();

