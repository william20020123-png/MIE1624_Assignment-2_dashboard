/* SalaryLens — the browser implementation of MIE1624 Assignment 2 Q5.4/Q5.5.
   No server, no inference API: all computations happen locally.
   Model weights and training-fitted transforms are exported from the original notebook. */

export function encodeResponses(model, answers) {
  const features = new Map();
  const put = (key, value) => features.set(key, Number(value));
  for (const q of model.questions) {
    const type = model.types[q];
    const value = Object.prototype.hasOwnProperty.call(answers, q) ? answers[q] : null;
    if (type === 'numeric') {
      const missing = value == null || value === '' || !Number.isFinite(Number(value));
      put(q, missing ? (model.medians[q] ?? 0) : Number(value));
      put(`${q}__Missing`, missing ? 1 : 0);
    } else if (type === 'single') {
      const item = value == null ? 'Not answered' : String(value);
      put(`${q}_${item}`, 1);
    } else {
      if (value === null) put(`${q}__Missing`, 1);
      if (Array.isArray(value)) {
        for (const option of value) put(`${q}__${option}`, 1);
      } else if (typeof value === 'string' && value !== '') {
        for (const option of value.split(';').map(s => s.trim()).filter(Boolean)) put(`${q}__${option}`, 1);
      }
    }
  }
  const languages = answers.LanguagesHaveWorkedWith;
  const langList = Array.isArray(languages) ? languages :
    (typeof languages === 'string' ? languages.split(';').map(s => s.trim()).filter(Boolean) : []);
  put('ENG_LanguagesCount', langList.length);
  const codingYears = answers.YearsCode, workYears = answers.WorkExp;
  const leadMissing = codingYears == null || workYears == null || codingYears === '' || workYears === '';
  put('ENG_CodingLead', leadMissing ? model.leadMedian : Math.max(0, Number(codingYears) - Number(workYears)));
  put('ENG_CodingLead_Missing', leadMissing ? 1 : 0);
  return model.selectedColumns.map((col) => {
    let val = features.get(col) ?? 0;
    if (model.scale[col]) val = (val - model.scale[col].mean) / model.scale[col].scale;
    return val;
  });
}
function sigmoid(x) {
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const y = Math.exp(x);
  return y / (1 + y);
}
export function predictProbabilities(model, answers) {
  const vec = encodeResponses(model, answers);
  let last = 0;
  const thresholds = model.coefficients.map((weights, k) => {
    let z = model.intercepts[k];
    for (let i = 0; i < weights.length; i++) z += weights[i] * vec[i];
    last = Math.max(last, sigmoid(z)); // Notebook: np.maximum.accumulate(cdf)
    return last;
  });
  const bounds = [0, ...thresholds, 1];
  const probs = bounds.slice(1).map((p, i) => Math.max(0, p - bounds[i]));
  const total = probs.reduce((a, b) => a + b, 0);
  return probs.map(p => p / total);
}
export function getRecommendations(model, answers, baseline = null) {
  const baseProbs = baseline || predictProbabilities(model, answers);
  const expected = baseProbs.reduce((s, p, i) => s + i * p, 0);
  const scores = [];
  for (const {question, technology} of model.skills) {
    const current = answers[question];
    if (current == null) continue; // Q5.5: cannot assume a skipped question means unskilled
    const currentList = Array.isArray(current) ? current :
      String(current).split(';').map(s => s.trim()).filter(Boolean);
    if (currentList.includes(technology)) continue;
    const simulated = {...answers, [question]: [...currentList, technology]};
    const after = predictProbabilities(model, simulated)
      .reduce((s, p, i) => s + i * p, 0);
    scores.push({technology, question, delta: after - expected});
  }
  return scores.sort((a, b) => b.delta - a.delta);
}

// UI / data binding. Runs only in a browser, keeping inference functions importable for verification.
if (typeof document !== 'undefined') {
  const byId = id => document.getElementById(id);
  const sectionNames = [
    ['Country', 'Age', 'Employment', 'EmploymentAddl', 'SOYears'],
    ['YearsCode', 'WorkExp', 'RoleFunctions'],
    ['LanguagesHaveWorkedWith', 'DatabasesHaveWorkedWith', 'CloudPlatHaveWorkedWith', 'CodeTestHaveWorkedWith', 'DevIDEHaveWorkedWith']
  ];
  const sections = [
    ['Personal background', 'Where you live and your current work or study status.'],
    ['Experience & activities', 'Your coding experience and typical weekly activities.'],
    ['Technologies & tools', 'What you actually used in the past year. Choose multiple where appropriate.']
  ];
  let model = null;
  let step = 0;
  const answers = {};
  const status = byId('status');
  const content = byId('question-container');
  const clearError = () => {status.hidden = true; status.textContent = '';};
  const showError = msg => {status.hidden = false; status.textContent = msg; byId('status').scrollIntoView({behavior:'smooth', block:'center'});};
  const niceTitles = {Country:'Country of residence',Age:'Age range',Employment:'Employment status',EmploymentAddl:'Additional responsibilities',SOYears:'Stack Overflow usage',YearsCode:'Years of coding experience',WorkExp:'Professional work experience',RoleFunctions:'Typical weekly activities',LanguagesHaveWorkedWith:'Programming languages',DatabasesHaveWorkedWith:'Databases',CloudPlatHaveWorkedWith:'Platforms & package managers',CodeTestHaveWorkedWith:'Testing & monitoring tools',DevIDEHaveWorkedWith:'Development environments'};
  function h(tag, className='', text=null) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== null) el.textContent = String(text);
    return el;
  }
  function fieldBase(q) {
    const outer = h('div','field');
    const lab = h('label','field-label');
    lab.append(h('span','',niceTitles[q] ?? q));
    lab.append(h('span','code',q));
    const question = h('p','field-question',model.questionText[q] ?? q);
    outer.append(lab,question);
    return outer;
  }
  function numericField(q) {
    const outer = fieldBase(q);
    const input = h('input');
    input.type='number'; input.step='1'; input.min='0'; input.max='120'; input.id=`answer-${q}`;
    input.placeholder = q==='WorkExp' ? 'Optional — leave blank if none / not answered' : 'Enter whole years';
    if (answers[q] != null) input.value = answers[q];
    input.addEventListener('input', () => {
      const v=input.value.trim(); answers[q] = v === '' ? null : Number(v);
      updateProgress();
    });
    outer.append(input);
    outer.append(h('p','field-bottom',q==='WorkExp' ? 'You may leave this blank, as allowed by the original survey.' : 'Use a whole number (0 or greater).'));
    return outer;
  }
  function singleField(q) {
    const outer = fieldBase(q), wrap = h('div','single-wrap');
    const select = h('select');select.id=`answer-${q}`;
    const placeholder = new Option('Choose an answer…','__CHOOSE__');
    select.add(placeholder);
    select.add(new Option('Not answered / prefer not to say','__SKIP__'));
    for(const answer of model.choices[q]) select.add(new Option(answer,answer));
    select.value = answers[q] === undefined ? '__CHOOSE__' : (answers[q] === null ? '__SKIP__' : String(answers[q]));
    select.addEventListener('change',()=>{answers[q] = select.value === '__CHOOSE__' ? undefined : (select.value === '__SKIP__' ? null : select.value); updateProgress();});
    wrap.append(select);outer.append(wrap);return outer;
  }
  function multiField(q) {
    const outer = fieldBase(q), wrap=h('div','multi-wrap'), header=h('div','multi-header');
    const search=h('input'); search.type='search';search.placeholder='Search answers…';search.setAttribute('aria-label',`Search ${niceTitles[q]}`);
    const selectedNumber=h('span','selection-summary');
    header.append(search,selectedNumber);
    const options=h('div','multi-options'), footer=h('div','multi-footer');
    const noneLabel=h('label'), skipLabel=h('label');
    const none=h('input');none.type='checkbox';none.checked=Array.isArray(answers[q]) && answers[q].length===0;
    const skip=h('input');skip.type='checkbox';skip.checked=answers[q]===null;
    noneLabel.append(none,h('span','','None used'));
    skipLabel.append(skip,h('span','','Not answered'));
    footer.append(noneLabel,skipLabel,h('small','','Select an answer, or explicitly choose none / skipped.'));
    wrap.append(header,options,footer);outer.append(wrap);
    function renderChoices() {
      options.replaceChildren();
      const chosen = Array.isArray(answers[q]) ? answers[q] : [];
      const needle=search.value.toLocaleLowerCase().trim();
      const found=model.choices[q].filter(x => x.toLocaleLowerCase().includes(needle));
      selectedNumber.textContent=`${chosen.length} selected`;
      none.checked=Array.isArray(answers[q]) && chosen.length===0;
      skip.checked=answers[q]===null;
      search.disabled=answers[q]===null;
      for(const item of found) {
        const lab=h('label','multi-option');
        const check=h('input'); check.type='checkbox';check.checked=chosen.includes(item);check.disabled=answers[q]===null;
        const text=h('span','',item);lab.append(check,text);
        check.addEventListener('change',()=>{
          const old=Array.isArray(answers[q])?answers[q]:[];
          answers[q]=check.checked?[...old,item]:old.filter(x=>x!==item);
          renderChoices();updateProgress();
        });
        options.append(lab);
      }
      if (!found.length) options.append(h('p','rec-empty','No matching options.'));
    }
    search.addEventListener('input',renderChoices);
    none.addEventListener('change',()=>{answers[q]=none.checked?[]:undefined;renderChoices();updateProgress();});
    skip.addEventListener('change',()=>{answers[q]=skip.checked?null:undefined;renderChoices();updateProgress();});
    renderChoices();return outer;
  }
  function updateProgress() {
    let completed=0;
    for (const q of model.questions) {
      const val=answers[q];
      if (val!==undefined) completed++;
    }
    const n=model.questions.length;
    byId('progress-fill').style.width=`${Math.round(100*completed/n)}%`;
    byId('progress-text').textContent=`${completed} of ${n} responses · ${Math.round(100*completed/n)}%`;
  }
  function renderStep(newStep) {
    step=newStep;clearError();
    document.querySelectorAll('.step-button').forEach(b=>b.classList.toggle('active',Number(b.dataset.step)===step));
    byId('section-heading').textContent=sections[step][0];
    byId('section-subheading').textContent=sections[step][1];
    byId('section-kicker').textContent=`SECTION 0${step+1} / 03`;
    const qs=sectionNames[step].filter(q=>model.questions.includes(q));
    byId('question-count').textContent=`${qs.length} questions`;
    content.replaceChildren();
    for(const q of qs){const type=model.types[q];content.append(type==='numeric'?numericField(q):type==='single'?singleField(q):multiField(q));}
    byId('back-btn').disabled=step===0;
    byId('next-btn').hidden=step===2;
    byId('predict-btn').hidden=step!==2;
    updateProgress();
  }
  function checkAnswers() {
    const missing=[];
    for (const q of model.questions) {
      if ((model.types[q]==='single' || model.types[q]==='multi') && answers[q]===undefined) missing.push(q);
      if (q==='YearsCode' && (!Number.isInteger(answers[q]) || answers[q]<0)) missing.push(q);
      if (q==='WorkExp' && answers[q]!=null && (!Number.isInteger(answers[q]) || answers[q]<0)) missing.push(q);
    }
    return missing;
  }
  function shortLabel(i) {
    const labels=['<$10k','$10–20k','$20–30k','$30–40k','$40–50k','$50–75k','$75–100k','$100–125k','$125–150k','$150–200k','$200–250k','$250–300k','$300–400k','$400–500k','>$500k'];
    return labels[i];
  }
  function renderResults() {
    const missing=checkAnswers();
    if (missing.length) {
      const foundStep=sectionNames.findIndex(qs=>qs.some(q=>missing.includes(q)));
      renderStep(Math.max(0,foundStep));
      showError(`Please complete the following required answers: ${[...new Set(missing)].map(q=>niceTitles[q]).join(', ')}. Use "Not answered" where appropriate.`);
      return;
    }
    const probs=predictProbabilities(model,answers);
    const sorted=probs.map((p,i)=>({p,i})).sort((a,b)=>b.p-a.p);
    const top=sorted[0];
    byId('most-likely').textContent=model.buckets[top.i];
    byId('most-confidence').textContent=`${(100*top.p).toFixed(1)}% probability`;
    byId('mean-bucket').textContent=probs.reduce((sum,p,i)=>sum+i*p,0).toFixed(2);
    byId('top3-prob').textContent=`${(sorted.slice(0,3).reduce((s,v)=>s+v.p,0)*100).toFixed(1)}%`;
    const chart=byId('probability-chart');chart.replaceChildren();
    const max=Math.max(...probs);
    probs.forEach((p,i)=>{
      const row=h('div',`prob-row${i===top.i?' highlight':''}`);
      const label=h('span','prob-label',`${String(i).padStart(2,'0')}  ${shortLabel(i)}`);label.title=model.buckets[i];
      const track=h('div','prob-track'),bar=h('span','prob-bar');bar.style.width=`${p>0?100*p/max:0}%`;track.append(bar);
      row.append(label,track,h('span','prob-value',`${(p*100).toFixed(1)}%`));chart.append(row);
    });
    const ranked=getRecommendations(model,answers,probs), rec=byId('recommendations'); rec.replaceChildren();
    if(!ranked.length){rec.append(h('p','rec-empty','No remaining selected technologies can be compared. Questions you marked as unanswered are excluded from recommendations.'));}
    else{
      const positive=ranked.filter(s=>s.delta>0);
      if(!positive.length)rec.append(h('p','rec-empty','None of the remaining technology additions predicts a positive expected-bucket shift; all candidate rankings are shown below.'));
      ranked.forEach((item,i)=>{
        const row=h('div','rec-item');
        const rank=h('span',`rec-rank${i<3 && item.delta>0?' top':''}`,String(i+1).padStart(2,'0'));
        const middle=h('div'); middle.append(h('span','rec-name',item.technology),h('span','rec-sub',niceTitles[item.question] ?? item.question));
        const diff=h('span',`rec-delta${item.delta>0?' positive':''}`,`${item.delta>=0?'+':''}${item.delta.toFixed(3)}`);
        row.append(rank,middle,diff);rec.append(row);
      });
    }
    byId('results').hidden=false;
    byId('results').scrollIntoView({behavior:'smooth',block:'start'});
  }
  byId('back-btn').addEventListener('click',()=>{if(step>0)renderStep(step-1);});
  byId('next-btn').addEventListener('click',()=>{if(step<2)renderStep(step+1);});
  byId('predict-btn').addEventListener('click',renderResults);
  byId('edit-btn').addEventListener('click',()=>{byId('results').hidden=true;byId('survey-form').scrollIntoView({behavior:'smooth'});});
  document.querySelectorAll('.step-button').forEach(b=>b.addEventListener('click',()=>renderStep(Number(b.dataset.step))));
  fetch('./model.json').then(response=>{
    if(!response.ok)throw Error(`Could not load model.json (HTTP ${response.status})`);
    return response.json();
  }).then(data=>{
    model=data;
    // Ask users explicitly to choose selected tools, "none used", or "not answered".
    renderStep(0);
  }).catch(error=>{
    content.append(h('p','rec-empty',`Unable to load the trained model: ${error.message}. Deploy all files together to GitHub Pages, or preview locally using a web server rather than opening index.html via file://.`));
    byId('next-btn').disabled=true;
    console.error(error);
  });
}
