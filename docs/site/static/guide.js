// Copy buttons on code blocks, and KaTeX auto-render for $...$ and $$...$$.
document.querySelectorAll("main pre").forEach(function (pre) {
  if (pre.closest(".codeblock")) return;
  var wrap = document.createElement("div");
  wrap.className = "codeblock";
  pre.parentNode.insertBefore(wrap, pre);
  wrap.appendChild(pre);
  var button = document.createElement("button");
  button.type = "button";
  button.className = "copy";
  button.textContent = "Copy";
  button.addEventListener("click", function () {
    var text = pre.innerText;
    var done = function () {
      button.textContent = "Copied";
      setTimeout(function () { button.textContent = "Copy"; }, 1500);
    };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done);
    } else {
      var area = document.createElement("textarea");
      area.value = text;
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      document.body.removeChild(area);
      done();
    }
  });
  wrap.appendChild(button);
});

if (window.renderMathInElement) {
  renderMathInElement(document.querySelector("main"), {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
    ],
  });
}
