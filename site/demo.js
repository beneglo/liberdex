/* The hero's replay. Nothing here searches: three recorded searches, the answer and the
   rows the ranker settled on, played back
   at roughly the pace the real thing runs. Every host ends in .example, a
   reserved name, because a real result page belongs to whoever published it
   and none of them is shown here. The passages are written for this page. */

window.LIBERDEX_DEMO = [
  {
    id: "asyncio",
    query: "asyncio cancel task python",
    tick: { firstUrl: 212, dispatched: 31, read: 27, rank: 118, total: 3480 },
    answer: {
      text: "Task.cancel() only requests cancellation: a CancelledError is thrown into the coroutine at its next await, and the task is finished only once you await it and let that error propagate. Swallow the error and the task keeps running.",
      cites: [1, 2],
    },
    hits: [
      { site: "docs.pyasync.example", source: "llm", rel: 0.97,
        title: "Coroutines and tasks: cancellation",
        url: "https://docs.pyasync.example/library/tasks.html#cancellation",
        text: "Task.cancel() arranges for a CancelledError to be thrown into the wrapped coroutine at the next await. The coroutine may clean up in a finally block and re-raise. Suppressing the error keeps the task alive, which is almost never what the caller wanted; check cancelled() afterwards if you must." },
      { site: "asyncnotes.example", source: "expand", rel: 0.91,
        title: "Cancel an asyncio task, then wait for it",
        url: "https://asyncnotes.example/cancel-a-task/",
        text: "Calling cancel() does not stop anything by itself. Until you await the task it is still running, and the exception you expect to see has not been raised yet. The pattern is cancel, then await inside a try that catches CancelledError, then move on." },
      { site: "forum.pyasync.example", source: "hub", rel: 0.84,
        title: "Cancelling tasks properly",
        url: "https://forum.pyasync.example/t/cancelling-tasks-properly",
        text: "The thread that keeps coming back: a task cancelled inside gather() takes its siblings with it unless return_exceptions is set, and a TaskGroup cancels the whole group on the first failure by design. Both are documented, neither is obvious the first time." },
      { site: "docs.pyasync.example", source: "sitemap", rel: 0.72,
        title: "asyncio.timeout(): cancellation with a deadline",
        url: "https://docs.pyasync.example/library/timeouts.html",
        text: "The context manager cancels the enclosed task when the deadline passes and turns the resulting CancelledError into a TimeoutError at the block's edge, so the code outside sees a timeout and the code inside sees a cancellation." },
    ],
  },
  {
    id: "zugspitze",
    query: "Zugspitze Höhe",
    tick: { firstUrl: 198, dispatched: 24, read: 22, rank: 96, total: 3120 },
    answer: {
      text: "Die Zugspitze ist mit 2.962 m der höchste Berg Deutschlands. Der Gipfel liegt auf der Grenze zwischen Bayern und Tirol.",
      cites: [1],
    },
    hits: [
      { site: "bergwelten.example", source: "llm", rel: 0.96,
        title: "Zugspitze: Deutschlands höchster Gipfel",
        url: "https://bergwelten.example/gipfel/zugspitze",
        text: "Die Zugspitze ist mit 2.962 Metern der höchste Berg Deutschlands und der höchste Gipfel des Wettersteingebirges. Über den Gipfel verläuft die Grenze zu Österreich; das Gipfelkreuz steht auf der deutschen Seite." },
      { site: "alpenverein.example", source: "expand", rel: 0.88,
        title: "Zugspitze: Anstiege, Hütten, Höhenangaben",
        url: "https://alpenverein.example/wissen/zugspitze",
        text: "Vier klassische Anstiege führen auf den Gipfel: durch das Reintal, über das Höllental, über die Wiener-Neustädter-Hütte und über den Jubiläumsgrat. Die Höhendifferenz vom Eibsee beträgt rund 2.000 Meter." },
      { site: "wetterdienst.example", source: "sitemap", rel: 0.74,
        title: "Messstation Zugspitze",
        url: "https://wetterdienst.example/station/zugspitze",
        text: "Die Station steht seit 1900 auf dem Gipfelplateau und ist die höchstgelegene Wetterstation Deutschlands. Jahresmitteltemperatur unter dem Gefrierpunkt, Schnee an mehr als 300 Tagen im Jahr." },
    ],
  },
  {
    id: "beijing",
    query: "北京 天气 预报",
    tick: { firstUrl: 205, dispatched: 19, read: 17, rank: 84, total: 2960 },
    answer: null,
    hits: [
      { site: "qixiang.example", source: "llm", rel: 0.95,
        title: "北京 天气预报",
        url: "https://qixiang.example/beijing/",
        text: "北京今日多云转晴，最高气温 26°C，最低 15°C，东北风 2 到 3 级。未来三天以晴到多云为主，早晚温差较大，注意添衣。" },
      { site: "tianqi.example", source: "expand", rel: 0.89,
        title: "北京一周天气",
        url: "https://tianqi.example/city/beijing/week",
        text: "本周北京以晴好天气为主，周四夜间有分散性小雨，周末气温回升至 28°C 左右。空气质量良，适宜户外活动。" },
      { site: "lvyou.example", source: "hub", rel: 0.66,
        title: "北京旅游最佳季节",
        url: "https://lvyou.example/beijing/best-season",
        text: "秋季是游览北京的最佳时节：九月到十月天高气爽，雨水少，香山红叶多在十月中下旬。夏季炎热多雨，冬季干冷。" },
    ],
  },
];
