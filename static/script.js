let ws;
let zoomLevel = 1;
let isDragging = false;
let startX, startY, translateX = 0, translateY = 0;

async function fetchModels() {
    try {
        const select = document.getElementById('modelSelect');
        const currentVal = select.value;
        const res = await fetch('/models');
        const data = await res.json();
        
        const standard = [
            { id: 'PEDiT-100M-FP16.pt', label: 'PEDiT-100M-FP16.pt (FP16 - Recommended)' },
            { id: 'PEDiT-100M-FP8.pt', label: 'PEDiT-100M-FP8.pt (FP8 - Fast & Light)' },
            { id: 'PEDiT-100M-INT8.pt', label: 'PEDiT-100M-INT8.pt (INT8 - Max Speed)' }
        ];
        
        const serverModels = (data && data.models) ? data.models : [];
        const added = new Set();
        select.innerHTML = '';
        
        // 1. Always add official standard models first
        standard.forEach(item => {
            const option = document.createElement('option');
            option.value = item.id;
            option.text = item.label;
            select.appendChild(option);
            added.add(item.id);
        });
        
        // 2. Add other local checkpoints
        let hasOther = false;
        serverModels.forEach(m => {
            if (!added.has(m)) {
                if (!hasOther) {
                    const sep = document.createElement('option');
                    sep.disabled = true;
                    sep.text = '── Other Checkpoints ──';
                    select.appendChild(sep);
                    hasOther = true;
                }
                const option = document.createElement('option');
                option.value = m;
                option.text = m;
                select.appendChild(option);
                added.add(m);
            }
        });
        
        if (currentVal && added.has(currentVal)) {
            select.value = currentVal;
        }
    } catch(e) {
        console.error("Error fetching models", e);
    }
}

function openModal(imgSrc) {
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');
    modalImg.src = imgSrc;
    modal.classList.add('active');
    
    // Reset zoom and pan
    zoomLevel = 1;
    translateX = 0;
    translateY = 0;
    updateTransform();
}

function closeModal() {
    const modal = document.getElementById('imageModal');
    modal.classList.remove('active');
}

function updateTransform() {
    const modalImg = document.getElementById('modalImage');
    modalImg.style.transform = `translate(${translateX}px, ${translateY}px) scale(${zoomLevel})`;
}

function zoomIn() {
    zoomLevel = Math.min(zoomLevel + 0.5, 5);
    updateTransform();
}

function zoomOut() {
    zoomLevel = Math.max(zoomLevel - 0.5, 0.5);
    updateTransform();
}

function zoomReset() {
    zoomLevel = 1;
    translateX = 0;
    translateY = 0;
    updateTransform();
}

function downloadModalImage() {
    const modalImg = document.getElementById('modalImage');
    const a = document.createElement('a');
    a.href = modalImg.src;
    a.download = `generation.jpg`;
    a.click();
}

// Drag functionality for zoomed image
const modalImg = document.getElementById('modalImage');
modalImg.addEventListener('mousedown', (e) => {
    if (zoomLevel > 1) {
        isDragging = true;
        startX = e.clientX - translateX;
        startY = e.clientY - translateY;
    }
});

window.addEventListener('mousemove', (e) => {
    if (isDragging) {
        translateX = e.clientX - startX;
        translateY = e.clientY - startY;
        updateTransform();
    }
});

window.addEventListener('mouseup', () => {
    isDragging = false;
});

window.addEventListener('wheel', (e) => {
    const modal = document.getElementById('imageModal');
    if (modal.classList.contains('active')) {
        e.preventDefault();
        if (e.deltaY < 0) {
            zoomIn();
        } else {
            zoomOut();
        }
    }
}, { passive: false });

function createFrameCard(step, imgSrc, modelTime, vaeTime, totalStepTime) {
    const card = document.createElement('div');
    card.className = 'frame-card';
    
    const imgContainer = document.createElement('div');
    imgContainer.className = 'frame-img-container';
    imgContainer.onclick = () => openModal("data:image/jpeg;base64," + imgSrc);
    
    const img = document.createElement('img');
    img.className = 'frame-img';
    img.src = "data:image/jpeg;base64," + imgSrc;
    
    const zoomIcon = document.createElement('div');
    zoomIcon.className = 'zoom-icon';
    zoomIcon.innerHTML = `<svg viewBox="0 0 24 24"><path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/><path d="M12 10h-2v2H9v-2H7V9h2V7h1v2h2v1z"/></svg>`;
    
    imgContainer.appendChild(img);
    imgContainer.appendChild(zoomIcon);
    
    const info = document.createElement('div');
    info.className = 'frame-info';
    
    const header = document.createElement('div');
    header.className = 'frame-header';
    
    const stepEl = document.createElement('div');
    stepEl.className = 'frame-step';
    stepEl.innerText = 'Step ' + step;
    
    const rightControls = document.createElement('div');
    rightControls.style.display = 'flex';
    rightControls.style.alignItems = 'center';
    rightControls.style.gap = '0.5rem';

    const dlBtn = document.createElement('button');
    dlBtn.className = 'icon-btn';
    dlBtn.title = 'Download';
    dlBtn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>`;
    dlBtn.onclick = (e) => {
        e.stopPropagation();
        const a = document.createElement('a');
        a.href = "data:image/jpeg;base64," + imgSrc;
        a.download = `step_${step}.jpg`;
        a.click();
    };
    
    const totalEl = document.createElement('div');
    totalEl.className = 'frame-total-time';
    totalEl.innerText = `${totalStepTime} ms`;
    
    rightControls.appendChild(totalEl);
    rightControls.appendChild(dlBtn);
    
    header.appendChild(stepEl);
    header.appendChild(rightControls);
    
    const timesEl = document.createElement('div');
    timesEl.className = 'frame-times';
    
    timesEl.innerHTML = `
        <div class="time-row"><span class="time-label">Model Generation:</span> <span class="time-val">${modelTime} ms</span></div>
        <div class="time-row"><span class="time-label">VAE Processing:</span> <span class="time-val">${vaeTime} ms</span></div>
    `;
    
    info.appendChild(header);
    info.appendChild(timesEl);
    
    card.appendChild(imgContainer);
    card.appendChild(info);
    
    return card;
}

function generate() {
    const model = document.getElementById('modelSelect').value;
    const device = document.getElementById('deviceSelect').value;
    const prompt = document.getElementById('promptInput').value;
    const width = document.getElementById('widthInput').value;
    const height = document.getElementById('heightInput').value;
    const steps = document.getElementById('stepsInput').value;
    const cfg = document.getElementById('cfgInput').value;
    const seed = document.getElementById('seedInput').value;
    const fastGen = document.getElementById('fastGenInput').checked;
    
    const btn = document.getElementById('generateBtn');
    const status = document.getElementById('statusText');
    const gallery = document.getElementById('gallery');
    const summaryBanner = document.getElementById('summaryBanner');
    
    gallery.innerHTML = '';
    summaryBanner.classList.remove('active');
    btn.disabled = true;
    status.innerText = "Connecting...";

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/generate`);
    
    ws.onopen = () => {
        status.innerText = "Loading model & preparing...";
        ws.send(JSON.stringify({
            model: model,
            device: device,
            prompt: prompt,
            width: width,
            height: height,
            steps: steps,
            cfg: cfg,
            seed: seed ? parseInt(seed) : null,
            fast_generation: fastGen
        }));
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.error) {
            status.innerText = "Error: " + data.error;
            btn.disabled = false;
            ws.close();
            return;
        }
        
        if (data.status === "downloading") {
            status.innerText = data.message || "Downloading model from Hugging Face...";
            return;
        }
        
        if (data.status === "starting") {
            status.innerText = "Generating frames...";
        } else if (data.status === "done") {
            status.innerText = "Generation complete.";
            btn.disabled = false;
            
            // Show summary banner
            document.getElementById('totalModelTime').innerText = data.total_model_time_ms + ' ms';
            document.getElementById('totalVaeTime').innerText = data.total_vae_time_ms + ' ms';
            document.getElementById('totalOverallTime').innerText = data.total_time_ms + ' ms';
            summaryBanner.classList.add('active');
            
            ws.close();
        } else if (data.step) {
            if (data.image) {
                const card = createFrameCard(
                    data.step, 
                    data.image, 
                    data.model_time_ms, 
                    data.vae_time_ms, 
                    data.step_time_ms
                );
                gallery.appendChild(card);
                
                // Scroll to bottom
                const container = document.querySelector('.gallery-container');
                container.scrollTop = container.scrollHeight;
                
                status.innerText = `Step ${data.step} rendered.`;
            } else {
                status.innerText = `Generating step ${data.step}...`;
            }
        }
    };
    
    ws.onclose = () => {
        btn.disabled = false;
        if(status.innerText === "Generating frames...") {
            status.innerText = "Connection closed.";
        }
    };
    
    ws.onerror = (e) => {
        console.error(e);
        status.innerText = "WebSocket error.";
        btn.disabled = false;
    };
}

window.onload = fetchModels;
