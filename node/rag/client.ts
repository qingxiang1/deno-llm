const port = 3001;

async function fetchStream() {
  const response = await fetch(`http://localhost:${port}`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify({
      question: "在国内，球状闪电的案例大概有多少",
      session_id: "test-server",
    }),
  });

  const reader = response?.body?.getReader();
  const decoder = new TextDecoder();

  while (true && reader) {
    const { done, value } = await reader.read();
    if (done) break;
    console.log(decoder.decode(value));
  }

  console.log("Stream has ended");
}

fetchStream();
