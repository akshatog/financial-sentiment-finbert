import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

// Rebuild with the bundled Codex runtime, or set ARTIFACT_TOOL_MODULE to the
// installed @oai/artifact-tool module. Source evidence always comes from outputs/.
const workspaceDir = path.resolve(import.meta.dirname, '..');
const runtimeRoot = process.env.CODEX_ARTIFACT_RUNTIME || 'C:/Users/aksha/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES ||= path.join(runtimeRoot,'node/node_modules');
process.env.RUNTIME_NODE ||= path.join(runtimeRoot,'node/bin/node.exe');
const SKILL_DIR = process.env.CODEX_PRESENTATION_SKILL || 'C:/Users/aksha/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const modulePath = process.env.ARTIFACT_TOOL_MODULE || path.join(runtimeRoot, 'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs');
const {Presentation, PresentationFile, FileBlob} = await import(pathToFileURL(modulePath));
const {applyPresentationChartFont, finalizePresentation} = await import(pathToFileURL(path.join(SKILL_DIR,'container_tools/artifact_tool_utils.mjs')));
const RUNTIME_PYTHON = path.join(runtimeRoot,'python/python.exe');
const buildDir = path.join(workspaceDir,'.cache/presentation');
const outDir = path.join(workspaceDir,'presentation');
await fs.mkdir(buildDir,{recursive:true});
const read = async rel => JSON.parse(await fs.readFile(path.join(workspaceDir, rel),'utf8'));
const audit = await read('outputs/dataset_audit.json');
const meta = await read('models/baseline_metadata.json');
const evaluation = await read('outputs/evaluation.json');
let finbert = null;
for(const file of ['outputs/finbert_diagnostic.json','outputs/finbert_evaluation.json','outputs/evaluation_finbert.json']){
  try {finbert=await read(file);break;}catch(e){if(e.code!=='ENOENT') throw e;}
}
const diagnostic = finbert?.finbert || finbert?.diagnostic || finbert || evaluation.finbert;
const B = evaluation.baseline;
const M = evaluation.majority_baseline;
if(!B || !M) throw new Error('Evaluation must contain baseline and majority_baseline.');
const main=audit.main_dataset;
const partitions=audit.splits.partitions;
const labels=['negative','neutral','positive'];
const count=(obj,label)=>Number(obj?.[label]||0);
const pct=n=>(n*100).toFixed(1)+'%';
// Eight decimal places preserve more precision than the displayed 0.1% labels
// and remain within Excel's numeric precision for embedded native-chart data.
const chartMetric=n=>Number(n.toFixed(8));
const fmt=n=>Number(n).toLocaleString('en-US');
const negativeF1=B.per_class.negative['f1-score'];
const cm=B.confusion_matrix;
const totalErrors=cm.counts.reduce((sum,row,i)=>sum+row.reduce((r,v,j)=>r+(i===j?0:v),0),0);
const ni=cm.labels.indexOf('neutral'),pi=cm.labels.indexOf('positive');
const positiveNeutralErrors=cm.counts[ni][pi]+cm.counts[pi][ni];
const cleanCounts=main.clean_class_counts;
const theme={navy:'#112B3D',ink:'#172E40',teal:'#007C83',muted:'#516779',pale:'#EDF4F6',line:'#CEDBE1',white:'#FFFFFF',amber:'#975700',red:'#B34B4B',gray:'#8094A3'};
const FONT='Arial';
const presentation=Presentation.create({slideSize:{width:1280,height:720}});
function text(slide, value, x,y,w,h,size=28,color=theme.ink,bold=false){
  const sh=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  sh.text=value;sh.text.style={typeface:FONT,fontSize:size,color,bold};sh.text.insets=0;return sh;
}
function slide(title,notes=''){
  const s=presentation.slides.add();s.background.fill=theme.white;
  text(s,title,72,54,1136,84,43,theme.ink,true);
  text(s,String(presentation.slides.items.length).padStart(2,'0'),1152,663,56,26,17,theme.muted);
  s.speakerNotes.text=notes;
  return s;
}
function para(s,heading,body,y,{x=72,w=1100,hs=29,bs=26}={}){
  text(s,heading,x,y,w,42,hs,theme.teal,true);
  text(s,body,x,y+47,w,112,bs,theme.ink);
}
function table(s,values,x,y,w,h,widths){
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
  t.styleOptions={headerRow:true,bandedRows:false};
  t.borders.assign({style:'solid',fill:theme.line,width:1});
  for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
    const cell=t.getCell(r,c);cell.fill=r===0?theme.navy:theme.white;
    cell.text.style={typeface:FONT,fontSize:r===0?24:25,color:r===0?theme.white:theme.ink,bold:r===0};
  }
  t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({margins:{left:16,right:16,top:13,bottom:13},anchor:'center'});
  return t;
}
function chart(s,options){
  const c=s.charts.add('bar',options);applyPresentationChartFont(c,{fontFamily:FONT});return c;
}
const sourceNotes='Sources: https://huggingface.co/ProsusAI/finbert ; https://huggingface.co/datasets/takala/financial_phrasebank ; https://arxiv.org/abs/1810.04805 ; https://arxiv.org/abs/1908.10063. Project evidence: outputs/dataset_audit.json, models/baseline_metadata.json, outputs/evaluation.json.';

{
const s=presentation.slides.add();s.background.fill=theme.navy;
text(s,'Financial News\nSentiment Analysis',72,88,1110,205,64,theme.white,true);
text(s,'FinBERT application with a trained TF-IDF baseline',76,318,1080,68,30,'#A6D9DC');
text(s,'Akshat Agarwal',76,421,1080,45,31,theme.white,true);
text(s,'Roll 202401100100028 · CS-5A',76,469,1080,40,25,'#A6D9DC');
text(s,'KIET Group of Institutions, Ghaziabad',76,526,1080,40,25,theme.white);
text(s,'Faculty: Kriti Sharma',76,573,1080,40,24,theme.white);
text(s,'Unit 4 · Deep Learning Project',76,630,1080,40,24,'#A6D9DC');
s.speakerNotes.text='The application classifies short English financial statements. The project trains the conventional baseline and uses downloaded FinBERT weights without additional transformer training. '+sourceNotes;
}
{
const s=slide('Financial sentiment classification', 'The examples are self-authored demonstrations with intended interpretations. They are not independent benchmark labels. Financial sentiment concerns the statement and entity described, rather than a guaranteed market reaction.');
text(s,'Input: a short English financial statement\nOutput: positive, negative, or neutral',72,154,1120,95,31);
table(s,[['Statement','Intended sentiment'],['Profit increased and the outlook improved.','Positive'],['The company expects an operating loss.','Negative'],['The annual report will appear on Friday.','Neutral']],72,294,1136,274,[866,270]);
text(s,'Sentiment describes financial language. It does not predict stock returns.',72,610,1100,45,25,theme.muted);
}
{
const s=slide('Application and model roles','Text and CSV input share validation and prediction logic. FinBERT loads once for reuse. The baseline stores TF-IDF and logistic regression in one fitted pipeline. '+sourceNotes);
table(s,[['Layer','Role'],['Streamlit interface','Text input, CSV upload, result inspection and export'],['Shared validation','Blank rows, input length and named class mapping'],['Pretrained FinBERT','Financial transformer inference with all three scores'],['Trained baseline','TF-IDF features and logistic regression']],72,165,1136,363,[320,816]);
text(s,'Both model routes return the same named labels and result schema.',72,574,1090,62,29,theme.teal,true);
}
{
const s=slide('Financial PhraseBank is imbalanced','The raw 50Agree subset has 4,846 rows: neutral 2,879, positive 1,363 and negative 604. This chart uses the cleaned canonical rows recorded by the preparation script. The agreement threshold concerns annotators. The other agreement subsets overlap and cannot act as independent test sets. '+sourceNotes);
chart(s,{position:{left:72,top:162,width:760,height:416},categories:['Negative','Neutral','Positive'],series:[{name:'Clean examples',values:labels.map(l=>count(cleanCounts,l)),fill:theme.teal}],hasLegend:false,barOptions:{direction:'column',gapWidth:95},xAxis:{textStyle:{fontSize:25,fill:theme.ink},line:{fill:theme.line,width:1}},yAxis:{min:0,numberFormatCode:'#,##0',textStyle:{fontSize:20,fill:theme.muted},majorGridlines:{fill:theme.line,width:1}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'#,##0',textStyle:{fontSize:24,fill:theme.ink,bold:true}},chartFill:'none',plotAreaFill:'none'});
text(s,fmt(main.clean_rows),900,195,290,82,64,theme.teal,true);
text(s,'clean examples',900,282,290,44,28);
text(s,'50Agree subset\nEnglish financial news\nThree sentiment labels',900,365,290,150,25,theme.muted);
text(s,'Macro F1 gives each class equal weight despite the neutral majority.',72,618,1120,44,25,theme.muted);
}
{
const s=slide('Data audit and saved partitions','Partition counts come directly from outputs/dataset_audit.json. Preparation handles normalized duplicate text before splitting. No fitted TF-IDF vocabulary sees validation or test text. Random partitions do not establish temporal or company-level independence. '+sourceNotes);
text(s,`${fmt(main.matching_duplicate_occurrences_removed)} matching duplicate occurrences removed\n${fmt(main.conflicting_rows_quarantined)} conflicting-label rows quarantined`,72,158,1130,105,30);
table(s,[['Partition','Examples','Purpose'],['Training',fmt(partitions.train.rows),'Fit vocabulary and classifier'],['Validation',fmt(partitions.validation.rows),'Select logistic regression settings'],['Test',fmt(partitions.test.rows),'Final held-out baseline metrics']],72,300,1136,270,[250,240,646]);
text(s,'Stratified 70/15/15 target split, seed 42. No normalized sentence overlap.',72,610,1136,45,24,theme.muted);
}
{
const s=slide('FinBERT and Unit 4 concepts','FinBERT is based on BERT base, with a three-class sequence classification head. The model card reports 12 encoder layers, hidden size 768 and 12 attention heads. The project uses pretrained weights without further fine-tuning. Its model configuration supplies named label mapping. '+sourceNotes);
para(s,'Tokenization','WordPiece tokens and special tokens convert each statement into model input IDs.',166);
para(s,'Contextual transformer encoder','BERT uses self-attention to construct representations that depend on surrounding tokens.',330);
para(s,'Classification head','Three logits become softmax scores. The largest named score selects a label.',494);
text(s,'This project uses pretrained FinBERT weights without further fine-tuning.',72,629,1110,40,24,theme.muted);
}
{
const s=slide('Baseline training','All candidates fit only the training split. The project chooses C using validation macro F1, then evaluates the selected saved pipeline on the held-out test partition. '+sourceNotes);
table(s,[['Component','Configuration'],['TF-IDF','Word unigrams and bigrams, sublinear term frequency'],['Vocabulary limits','Minimum document frequency 2, maximum 0.98'],['Logistic regression','Balanced classes, maximum 2,000 iterations'],['Validation search','C in 0.5, 1, 2 and 4']],72,170,1136,326,[330,806]);
text(s,`Selected C = ${meta.selected_C}    Vocabulary = ${fmt(meta.feature_count)} features`,72,552,1110,50,31,theme.teal,true);
text(s,'The saved pipeline keeps its fitted vocabulary and classifier together.',72,615,1110,45,25,theme.muted);
}
{
const s=slide('Text and CSV demonstration','Live demo: analyse a positive financial example, compare the baseline, then upload data/examples/demo_news.csv and download results. The CSV includes an intentionally blank row. Scores are model outputs rather than calibrated guarantees. FinBERT rejects text exceeding 512 tokenizer tokens including special tokens.');
let imagePath=null;for(const p of ['docs/screenshots/app.jpg','docs/screenshots/app.png','docs/screenshots/analyze.png','docs/screenshots/app_text.png']){try{await fs.access(path.join(workspaceDir,p));imagePath=path.join(workspaceDir,p);break;}catch{}}
if(imagePath){
  const bytes=await fs.readFile(imagePath);s.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:imagePath.endsWith('.jpg')?'image/jpeg':'image/png',alt:'Financial sentiment application running locally',fit:'contain',position:{left:72,top:154,width:430,height:464}});
  text(s,'Single text',570,180,638,45,29,theme.teal,true);text(s,'Inspect the label and all three model scores.',570,230,638,116,28);
  text(s,'Batch CSV',570,369,638,45,29,theme.teal,true);text(s,'Choose the text column and export each result with its row status.',570,419,638,150,28);
  text(s,'Live app',570,528,638,38,26,theme.teal,true);
  const liveLabel='financial-sentiment-akshatog.streamlit.app';
  const liveText=text(s,liveLabel,570,568,638,44,24,theme.ink);
  liveText.text.get(liveLabel).link={uri:'https://financial-sentiment-akshatog.streamlit.app/',isExternal:true};
}else{
  para(s,'Single text','Analyse a financial sentence. Inspect the predicted label and all three model scores.',178);
  para(s,'Batch CSV','Choose a text column. Inspect row status and download predictions with the original columns.',365);
}
text(s,'CSV limits: 2 MB and 500 rows. Blank or invalid rows keep a visible status.',72,629,1136,40,24,theme.muted);
s.speakerNotes.text+=' Live hosted application: https://financial-sentiment-akshatog.streamlit.app/ . Source repository: https://github.com/akshatog/financial-sentiment-finbert .';
}
{
const s=slide('Held-out baseline results',`Test partition: ${B.n_examples} examples. Both displayed models use exactly the same held-out partition. The majority predictor always returns the most common training label. Raw metrics: ${JSON.stringify({baseline:{accuracy:B.accuracy,macro_f1:B.macro_f1},majority:{accuracy:M.accuracy,macro_f1:M.macro_f1}})}. `+sourceNotes);
chart(s,{position:{left:72,top:166,width:1116,height:387},categories:['Accuracy','Macro F1'],series:[{name:'Majority baseline',values:[M.accuracy,M.macro_f1].map(chartMetric),fill:theme.gray},{name:'TF-IDF + logistic regression',values:[B.accuracy,B.macro_f1].map(chartMetric),fill:theme.teal}],hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{fontSize:23,fill:theme.ink}},barOptions:{direction:'column',grouping:'clustered',gapWidth:110},xAxis:{textStyle:{fontSize:26,fill:theme.ink},line:{fill:theme.line,width:1}},yAxis:{min:0,max:1,majorUnit:.2,numberFormatCode:'0%',textStyle:{fontSize:21,fill:theme.muted},majorGridlines:{fill:theme.line,width:1}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.0%',textStyle:{fontSize:25,fill:theme.ink,bold:true}},chartFill:'none',plotAreaFill:'none'});
text(s,`${fmt(B.n_examples)} test examples. Baseline training and selection exclude these rows.`,72,609,1136,50,26,theme.muted);
}
{
const s=slide('Baseline errors by class','The confusion matrix uses actual labels in rows and predicted labels in columns. It comes directly from outputs/evaluation.json. Per-class performance must be considered alongside overall accuracy. '+sourceNotes);
const order=cm.labels;table(s,[['Actual / predicted',...order.map(l=>l[0].toUpperCase()+l.slice(1))],...order.map((l,i)=>[l[0].toUpperCase()+l.slice(1),...cm.counts[i].map(String)])],72,184,840,328,[285,185,185,185]);
text(s,'Class F1',959,191,245,50,29,theme.teal,true);
text(s,labels.map(l=>`${l[0].toUpperCase()+l.slice(1)}  ${pct(B.per_class[l]['f1-score'])}`).join('\n\n'),959,262,245,239,27);
text(s,`${positiveNeutralErrors} of ${totalErrors} errors confuse positive and neutral statements.`,72,579,1120,52,27,theme.muted);
}
{
const s=slide('FinBERT diagnostic and prior exposure','The ProsusAI/finbert model card explicitly states Financial PhraseBank sentiment fine-tuning. Our local split cannot establish that any PhraseBank row was unseen by the published checkpoint. Therefore this metric is a diagnostic with prior training exposure and is not a fair unseen-data comparison against the trained baseline. '+sourceNotes);
if(diagnostic && Number.isFinite(diagnostic.accuracy)){
text(s,pct(diagnostic.accuracy),72,167,500,98,76,theme.teal,true);
text(s,`PhraseBank diagnostic accuracy\nMacro F1: ${pct(diagnostic.macro_f1)} on ${fmt(diagnostic.n_examples||B.n_examples)} examples`,78,279,1060,90,29);
}else{
text(s,'FinBERT uses a checkpoint already\nfine-tuned on Financial PhraseBank',72,169,1100,130,43,theme.teal,true);
}
text(s,'Prior training exposure affects this result',72,402,1110,60,35,theme.amber,true);
text(s,'A new local split cannot remove exposure during the checkpoint’s original training.\nThirty source-linked sentences await human label review.',72,483,1110,126,29);
}
{
const s=slide('Limitations and next experiment','Future work needs independently sourced financial sentences, human labels frozen before predictions, overlap checks, and evaluation once. FinBERT weights use CC BY-NC-SA 4.0 and PhraseBank CC BY-NC-SA 3.0, according to their current cards. Original project code has a separate MIT license. '+sourceNotes);
para(s,'Input limits','English financial sentences are the intended domain. Mixed statements and missing context can change the label.',170);
para(s,'Evidence limits','Random PhraseBank partitions do not establish performance on future news. Model scores lack calibration.',340);
para(s,'Next experiment','Freeze source-linked human labels on new financial statements, then evaluate both models once.',510);
}

// Save complete native objects and render each slide for visual inspection.
for(const [i,s] of presentation.slides.items.entries()){
  const blob=await presentation.export({slide:s,format:'png',scale:1});
  await fs.writeFile(path.join(buildDir,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
if(process.env.DECK_RENDER_ONLY==='1'){
  console.log('Rendered 12 evidence-based slide previews. Final export was not requested in this run.');
  process.exit(0);
}
const buildRun = new Date().toISOString().replaceAll(':','-').replaceAll('.','-');
const candidatePath=path.join(buildDir,`candidate-${buildRun}.pptx`);
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const finalPath=path.join(outDir,process.env.FINAL_DECK_NAME||'financial_sentiment.pptx');
const result=await finalizePresentation({workspaceDir,candidatePath,finalPath,pythonExecutable:RUNTIME_PYTHON,integrityValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[2,3,5,7,10].flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[2,3,5,7,10],requiredNativeChartOwnerSlides:[4,9],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(buildDir,`validation-${buildRun}.json`)});
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(const [i,s] of finalDeck.slides.items.entries()){
  const blob=await finalDeck.export({slide:s,format:'png',scale:1});
  await fs.writeFile(path.join(buildDir,`final-slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
console.log(JSON.stringify({finalPath,result},null,2));



