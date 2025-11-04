window.ResultsUI = (function() {
  const grid = () => $("#results-grid");
  const statusBox = () => $("#upload-status");
  const getPlmForm = () => ({
    endpoint: $("#plm-endpoint").val() || "",
    apikey: $("#plm-apikey").val() || "",
    model: $("#plm-model").val() || ""
  });
  const setPlmForm = (cfg) => {
    if (!cfg) return;
    $("#plm-endpoint").val(cfg.endpoint || "");
    $("#plm-apikey").val(cfg.apikey || "");
    $("#plm-model").val(cfg.model || "");
  };
  const getModelForm = () => ({
    engine: $("#model-engine").val() || "",
    note: $("#model-note").val() || "",
    // weights filename only for display purpose
    weightsName: $("#model-weights")[0] && $("#model-weights")[0].files[0] ? $("#model-weights")[0].files[0].name : ""
  });
  const setModelForm = (cfg) => {
    if (!cfg) return;
    $("#model-engine").val(cfg.engine || "yolov5");
    $("#model-note").val(cfg.note || "");
  };

  function renderCard(item) {
    const reliability = item.reliability || {};
    const warnings = (reliability.warnings || []).join("; ");
    const det = (typeof item.num_detections === "number") ? item.num_detections : (reliability.num_detections || "-");
    const rendered = item.rendered || item.rendered_url || null;
    return `
      <div class="col">
        <div class="card h-100 shadow-sm">
          <div class="ratio ratio-4x3 bg-light">
            ${rendered ? `<img src="${rendered}" alt="result" class="img-fluid object-fit-contain p-1"/>` : `<div class="d-flex align-items-center justify-content-center text-muted">无渲染图</div>`}
          </div>
          <div class="card-body py-2">
            <div class="d-flex justify-content-between align-items-center">
              <span class="small">检测数: <strong>${det}</strong></span>
              <span class="badge text-bg-${reliability.status === 'ok' ? 'success' : 'secondary'}">${(reliability.status || "-")}</span>
            </div>
            ${warnings ? `<div class="small text-warning mt-1">${warnings}</div>` : ""}
            ${item.json ? `<div class="mt-1"><a class="link-primary small" href="${item.json}" target="_blank">查看JSON</a></div>` : ""}
          </div>
        </div>
      </div>
    `;
  }

  function appendResults(items) {
    if (!items || !items.length) return;
    const html = items.map(renderCard).join("");
    grid().prepend(html);
  }

  function refreshResults() {
    $.getJSON("/api/results").done(function(res) {
      grid().empty();
      appendResults(res.items || []);
    });
  }

  function clearResults() {
    if (!confirm("确定要清空所有检测结果吗？此操作不可撤销。")) return;
    statusBox().text("正在清空结果...");
    $("#btn-clear-results").prop("disabled", true);
    $.ajax({
      url: "/api/clear-results",
      type: "POST"
    }).done(function(res) {
      statusBox().text(`已删除 ${res.removed || 0} 个文件`);
      refreshResults();
    }).fail(function(xhr) {
      const msg = (xhr.responseJSON && xhr.responseJSON.error) || "清空失败";
      statusBox().text(msg);
    }).always(function() {
      $("#btn-clear-results").prop("disabled", false);
    });
  }

  function upload(files) {
    const fd = new FormData();
    for (let i = 0; i < files.length; i++) {
      fd.append("files", files[i]);
    }
    statusBox().text("上传中并检测，请稍候...");
    $("#btn-upload").prop("disabled", true);

    $.ajax({
      url: "/api/upload",
      type: "POST",
      data: fd,
      processData: false,
      contentType: false
    }).done(function(res) {
      statusBox().text("检测完成");
      appendResults(res.results || []);
    }).fail(function(xhr) {
      const msg = (xhr.responseJSON && xhr.responseJSON.error) || "上传失败";
      statusBox().text(msg);
    }).always(function() {
      $("#btn-upload").prop("disabled", false);
    });
  }

  function bind() {
    $("#btn-upload").on("click", function() {
      const files = $("#file-input")[0].files;
      if (!files || !files.length) {
        statusBox().text("请选择至少一张图片");
        return;
      }
      upload(files);
    });

    $("#btn-clear-results").on("click", function() {
      clearResults();
    });

    // Sidebar: PLM/DeepSeek
    $("#btn-plm-save").on("click", function() {
      const cfg = getPlmForm();
      localStorage.setItem("plmSettings", JSON.stringify(cfg));
      statusBox().text("PLM/DeepSeek 设置已保存（本地）");
    });
    $("#btn-plm-test").on("click", function() {
      const cfg = getPlmForm();
      if (!cfg.endpoint || !cfg.apikey) {
        statusBox().text("请填写 Endpoint 与 API Key");
        return;
      }
      // Placeholder test
      statusBox().text("已读取到配置（演示）：" + cfg.endpoint);
    });

    // Sidebar: Model management
    $("#btn-model-apply").on("click", function() {
      const cfg = getModelForm();
      localStorage.setItem("modelSettings", JSON.stringify(cfg));
      statusBox().text(`模型设置已保存（本地）：${cfg.engine}${cfg.weightsName ? '，文件：' + cfg.weightsName : ''}`);
    });
    $("#btn-model-reset").on("click", function() {
      localStorage.removeItem("modelSettings");
      setModelForm({ engine: "yolov5", note: "" });
      $("#model-weights").val("");
      statusBox().text("已恢复默认模型设置（仅界面）");
    });
  }

  function init() {
    bind();
    // load saved settings
    try {
      const plm = JSON.parse(localStorage.getItem("plmSettings") || "null");
      setPlmForm(plm);
    } catch (e) {}
    try {
      const model = JSON.parse(localStorage.getItem("modelSettings") || "null");
      setModelForm(model);
    } catch (e) {}
    refreshResults();
  }

  return { init };
})();


