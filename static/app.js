const detectFile = document.getElementById("detect-file");
const detectBtn = document.getElementById("detect-btn");
const detectResult = document.getElementById("detect-result");
const detectImage = document.getElementById("detect-image");

const compareA = document.getElementById("compare-a");
const compareB = document.getElementById("compare-b");
const compareBtn = document.getElementById("compare-btn");
const compareResult = document.getElementById("compare-result");

function showResult(el, text, isError = false) {
    el.textContent = text;
    el.className = isError ? "result error" : "result";
    el.hidden = false;
}

detectBtn.addEventListener("click", async () => {
    if (!detectFile.files.length) return;

    detectBtn.disabled = true;
    showResult(detectResult, "Processing...");

    const form = new FormData();
    form.append("file", detectFile.files[0]);

    try {
        const res = await fetch("/api/detect", { method: "POST", body: form });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.detail || "Detection failed");
        }
        showResult(detectResult, `Faces: ${data.total_faces}`);
        detectImage.src = data.annotated_image;
        detectImage.hidden = false;
    } catch (err) {
        showResult(detectResult, err.message, true);
        detectImage.hidden = true;
    } finally {
        detectBtn.disabled = false;
    }
});

compareBtn.addEventListener("click", async () => {
    if (!compareA.files.length || !compareB.files.length) return;

    compareBtn.disabled = true;
    showResult(compareResult, "Processing...");

    const form = new FormData();
    form.append("image_a", compareA.files[0]);
    form.append("image_b", compareB.files[0]);

    try {
        const res = await fetch("/api/compare", { method: "POST", body: form });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.detail || "Comparison failed");
        }
        showResult(compareResult, `${data.verdict} (score: ${data.similarity})`);
    } catch (err) {
        showResult(compareResult, err.message, true);
    } finally {
        compareBtn.disabled = false;
    }
});
