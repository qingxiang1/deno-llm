import { readFileSync } from "fs";
import path from "path";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatAlibabaTongyi } from "@langchain/community/chat_models/alibaba_tongyi";
import { StringOutputParser } from "@langchain/core/output_parsers";
import readline from "readline";
import util from "util";
import "dotenv/config";

const guaInfoBuffer = readFileSync(path.join(__dirname, "./fortune.json"));
const guaInfo = JSON.parse(guaInfoBuffer.toString());

const yaoName = ["初爻", "二爻", "三爻", "四爻", "五爻", "六爻"];

/**
 * 通俗来讲，进行六爻占卜需要的三枚硬币，每次丢三枚硬币叫生成卦象的一部分，一共丢六次，称为六爻。
 * 每次丢硬币的时候，如果正面数量比背面多，那就是阳；背面比正面多，就是阴。每三个阴阳就能组成八卦中的一卦，
 * 两个卦就能对应八八六十四卦中的一个卦象，也就有对应的解读。
 */

/**
 * 穷举了八卦的所有情况
 */
const guaDict: any = {
  阳阳阳: "乾",
  阴阴阴: "坤",
  阴阳阳: "兑",
  阳阴阳: "震",
  阳阳阴: "巽",
  阴阳阴: "坎",
  阳阴阴: "艮",
  阴阴阳: "离",
};

function generateGua(): string[] {
  let yaoCount = 0;
  const messageList = [];

  /**
   * 算卦流程实现
   *
   * 八卦和八卦对应的信息属于是有真实答案的类别，一般这种类别不要让 llm 自己生成，大家可以测试一下，一般会输出一些不存在的卦象和解读。
   * 所以，类似于 RAG 的思路，我们把标准的算卦流程和真实的八卦信息，由我们代码生成，并在后续 chat 中，直接嵌入到 llm 上下文中。
   * 具体的实现过程就是把算卦流程编码化，写起来比较繁琐，但逻辑很简单。
   * 首先，我们定义一个生成 “一次爻” 的函数
   */
  const genYao = () => {
    const coinRes = Array.from({ length: 3 }, () =>
      Math.random() > 0.5 ? 1 : 0
    );
    const yinYang = coinRes.reduce((a, b): any => a + b, 0) > 1.5 ? "阳" : "阴";
    const message = `${yaoName[yaoCount]} 为 ${coinRes
      .map((i) => (i > 0.5 ? "字" : "背"))
      .join("")} 为 ${yinYang}`;

    return {
      yinYang,
      message,
    };
  };

  /**
   * 模拟算卦的流程：
   */
  const firstGuaYinYang = Array.from({ length: 3 }, () => {
    const { yinYang, message } = genYao();
    yaoCount++;

    messageList.push(message);
    return yinYang;
  });
  const firstGua = guaDict[firstGuaYinYang.join("")];
  messageList.push(`您的首卦为 ${firstGua}`);

  const secondGuaYinYang = Array.from({ length: 3 }, () => {
    const { yinYang, message } = genYao();
    yaoCount++;

    messageList.push(message);
    return yinYang;
  });
  const secondGua = guaDict[secondGuaYinYang.join("")];
  messageList.push(`您的次卦为 ${secondGua}`);

  const gua = secondGua + firstGua;
  const guaDesc = guaInfo[gua];

  const guaRes = `
    六爻结果: ${gua}  
    卦名为：${guaDesc.name}   
    ${guaDesc.des}   
    卦辞为：${guaDesc.sentence}   
  `;

  messageList.push(guaRes);

  return messageList;
}

// generateGua();

async function main() {
  const messageList = generateGua();

  const history = new ChatMessageHistory();
  const guaMessage = messageList.map((message): ["ai", string] => [
    "ai",
    message,
  ]);

  /**
   * 把代码生成的算卦信息，作为 ai 输出的内容，嵌入到 prompt 中：
   */
  const prompt = await ChatPromptTemplate.fromMessages([
    [
      "system",
      `你是一位出自中华六爻世家的卜卦专家，你的任务是根据卜卦者的问题和得到的卦象，为他们提供有益的建议。
        你的解答应基于卦象的理解，同时也要尽可能地展现出乐观和积极的态度，引导卜卦者朝着积极的方向发展。
        你的语言应该具有仙风道骨、雅致高贵的气质，以此来展现你的卜卦专家身份。`,
    ],
    ...guaMessage,
    new MessagesPlaceholder("history_message"), // 留出 history_message 的 place holder，方便后续插入历史聊天记录
    ["human", "{input}"],
  ]);

  const llm = new ChatAlibabaTongyi({
    model: "qwen-max",
  });
  const chain = prompt.pipe(llm).pipe(new StringOutputParser());
  
  /**
   * 使用 RunnableWithMessageHistory 去给 chain 添加历史聊天记录的能力
   */
  const chainWithHistory = new RunnableWithMessageHistory({
    runnable: chain,
    getMessageHistory: (_sessionId) => history,
    inputMessagesKey: "input",
    historyMessagesKey: "history_message",
  });

  /**
   * 使用 node 内置的 readline 模块去实现在 cli 的交互：
   */
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const question = util.promisify(rl.question).bind(rl);

  const input = await question("告诉我你的疑问: ");

  /**
   * 实用小技巧
   *
   * 1、使用 util.promisify 去生成了一个 promise 化的 question，方面我们使用 async await 的风格进行异步的编程。
   * 2、因为 llm 的返回比较耗时，所以用已经生成的算法流程的 messageList，去创建printMessagesPromise，会以 1s 的间隔去打印算卦的流程，
   *    让用户无形中等待 llm 请求的返回。
   */
  let index = 0;
  const printMessagesPromise = new Promise<void>((resolve) => {
    const intervalId = setInterval(() => {
      if (index < messageList.length) {
        console.log(messageList[index]);
        index++;
      } else {
        clearInterval(intervalId);
        resolve();
      }
    }, 1000);
  });

  const llmResPromise = chainWithHistory.invoke(
    { input: "用户的问题是：" + input },
    { configurable: { sessionId: "no-used" } }
  );

  const [_, firstRes] = await Promise.all([
    printMessagesPromise,
    llmResPromise,
  ]);

  console.log(firstRes);

  async function chat() {
    const input: any = await question("User: ");

    if (input?.toLowerCase() === "exit") {
      rl.close();
      return;
    }

    const response = await chainWithHistory.invoke(
      { input },
      { configurable: { sessionId: "no-used" } }
    );

    console.log("AI: ", response);
    chat();
  }

  chat();
}

main();
