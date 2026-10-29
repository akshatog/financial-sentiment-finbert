// Editable six-slide adaptation of a user-provided PowerPoint reference.
// The template is an external design input, not part of the runtime app.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const RUNTIME=process.env.CODEX_ARTIFACT_RUNTIME||'C:/Users/aksha/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const SKILL='C:/Users/aksha/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const templatePath=process.argv[2]||process.env.REFERENCE_DECK;
if(!templatePath)throw new Error('Supply the original reference PPTX as the first argument or REFERENCE_DECK.');
const run=new Date().toISOString().replace(/[:.]/g,'-');
const build=path.join(ROOT,'.cache/presentation/reference-revision');
const finalDir=path.join(ROOT,'.cache/presentation',`reference-final-${run}`);
await fs.mkdir(build,{recursive:true});await fs.mkdir(finalDir,{recursive:true});
process.env.RUNTIME_NODE_MODULES=path.join(RUNTIME,'node/node_modules');
process.env.RUNTIME_NODE=path.join(RUNTIME,'node/bin/node.exe');
const {PresentationFile,FileBlob}=await import(pathToFileURL(path.join(RUNTIME,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')));
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')));
const metrics=JSON.parse(await fs.readFile(path.join(ROOT,'outputs/evaluation.json'),'utf8'));
const audit=JSON.parse(await fs.readFile(path.join(ROOT,'outputs/dataset_audit.json'),'utf8'));
const deck=await PresentationFile.importPptx(await FileBlob.load(templatePath));
if(deck.slides.items.length!==6)throw new Error('Expected the inspected six-slide reference.');
const inspect=await deck.inspect({select:'$.slides[*]',include:['id','index','elements.id','elements.kind','elements.text','elements.position','elements.style']});
const records=inspect.records;
const originals=new Map(records.flatMap(slide=>slide.elements.map(el=>[el.id,el])));
const resolve=id=>deck.resolve(id);
function paras(id,values){
  const old=originals.get(id).text.split('\n'),shape=resolve(id);
  if(old.length!==values.length)throw new Error(`Paragraph count mismatch at ${id}`);
  for(let i=old.length-1;i>=0;i--)if(old[i]!==values[i])shape.text.replace(old[i],values[i]);
  return shape;
}
function text(id,value){const shape=resolve(id);shape.text=value;return shape;}
function labeled(id,prefix,value){
  const old=originals.get(id).text;
  if(!old.startsWith(prefix))throw new Error(`Expected ${prefix} at ${id}`);
  resolve(id).text.replace(old.slice(prefix.length),value);
}
function table(id,values){
  const target=resolve(id);
  for(let r=0;r<values.length;r++)for(let c=0;c<values[r].length;c++){
    target.cells.set(r,c,values[r][c]);
    target.getCell(r,c).text.style={typeface:'Calibri',fontSize:16,color:r===0?'#FFFFFF':'#1B2A2F',bold:r===0||c===0};
  }
}
const pct=n=>`${(n*100).toFixed(2)}%`;
const live='https://financial-sentiment-akshatog.streamlit.app/';
const repo='https://github.com/akshatog/financial-sentiment-finbert';

// Preserve the actual reference cover panels and left/right composition.
const title=text('sh/547294r6','Financial News\nSentiment Analysis');
title.position={left:57.6,top:96,width:403.2,height:134.4};
title.text.style={typeface:'Calibri',fontSize:48,bold:true,color:'#FFFFFF',alignment:'left'};
const subtitle=text('sh/k3yl0zql','Using FinBERT');
subtitle.position={left:57.6,top:244.8,width:403.2,height:57.6};
labeled('sh/65g3298r','Task: ','Classifies short English financial text as positive, negative or neutral.');
labeled('sh/sryl4zqx','Model: ','Pretrained FinBERT, with a separately trained TF-IDF + logistic regression baseline.');
labeled('sh/utg3698n','Stack: ','Python, Transformers, PyTorch, scikit-learn and Streamlit.');
labeled('sh/hofulsf2','Objective: ','Build a usable sentiment app and evaluate its results with clear training-data limitations.');
const identity=deck.slides.items[0].shapes.add({geometry:'textbox',position:{left:57.6,top:340.8,width:403.2,height:134.4},fill:'none',line:{fill:'none',width:0}});
identity.text='Akshat Agarwal\nRoll No. 202401100100028, CS-5A\nFaculty: Kriti Sharma\nKIET Group of Institutions, Ghaziabad';
identity.text.style={typeface:'Calibri',fontSize:17.33,color:'#FFFFFF'};identity.text.insets=0;

// Reuse all five native flow nodes and explanation panels.
text('sh/b29kza94','Financial Text');text('sh/ove9o7yd','BERT Encoder');
text('sh/nu58f2hs','Classification Head');text('sh/ahwrqhgj','Sentiment Label');
paras('sh/cnupgny5',['Tokenization','Converts financial text into WordPiece token IDs.']);
paras('sh/t8byxkn2',['Encoder','Uses self-attention to represent words in their surrounding context.']);
paras('sh/76twva5c',['Classification head','Maps the encoded statement to three sentiment logits.']);
paras('sh/hcvy14ne',['Pretrained weights','FinBERT already used PhraseBank in fine-tuning. We perform inference without further fine-tuning.']);
paras('sh/fadgzu58',['Inference','Softmax produces three model scores. The highest named score selects the sentiment label.']);

text('sh/cza94vmx','Training Pipeline: Three Model Routes');
table('tb/juho7qhg',[
  ['Setting','Majority baseline','TF-IDF + logistic regression','Pretrained FinBERT'],
  ['Representation','Class frequency','Word unigrams + bigrams','Contextual BERT features'],
  ['Training here','Training-label counts','3,385 training rows','No project fine-tuning'],
  ['Selection','Neutral majority','C = 2 on 725 validation rows','Published checkpoint'],
  ['Main purpose','Simple reference','Trained baseline','Main application model'],
]);
paras('sh/3ah8rqlg',['Majority to trained baseline','Learns word patterns from labeled data. The TF-IDF vocabulary fits on the training partition only.']);
paras('sh/pcjqtg36',['Baseline to FinBERT','Uses contextual pretrained representations. Its original PhraseBank exposure changes how we interpret the score.']);
text('sh/m5cra54z',`PhraseBank: ${audit.main_dataset.clean_rows.toLocaleString('en-US')} clean rows. Train / validation / test: 3,385 / 725 / 726. Seed 42.`);

// Keep the result cards and use the actual classification metrics.
text('sh/rm1k7yt4','Majority baseline');
paras('sh/ql8jytsj',['Accuracy',pct(metrics.majority_baseline.accuracy)]);
paras('sh/doj29oba',['Macro F1',pct(metrics.majority_baseline.macro_f1)]);
text('sh/3ihk3et8','Simple reference');
text('sh/kbm987y5','Trained TF-IDF + LR');
paras('sh/5cva1cfq',['Accuracy',pct(metrics.baseline.accuracy)]);
paras('sh/i94r6xgz',['Macro F1',pct(metrics.baseline.macro_f1)]);
text('sh/w72947yt','Held-out test result');
text('sh/a5kr2xg3','Pretrained FinBERT');
paras('sh/v6tsv2xo',['Accuracy',pct(metrics.finbert.accuracy)]);
paras('sh/g36tgryd',['Macro F1',pct(metrics.finbert.macro_f1)]);
text('sh/p0batw72','Prior-exposure diagnostic');
for(const el of records[3].elements.filter(el=>el.kind==='shape')){
  if(Math.abs(el.position.top-273.6)<.1&&Math.abs(el.position.left-67.2)<.1)resolve(el.id).fill='#FFFFFF';
  if(Math.abs(el.position.top-273.6)<.1&&Math.abs(el.position.left-662.4)<.1)resolve(el.id).fill='#F2A541';
}
const interpretation=resolve('sh/oz29krqh');
const oldBullets=originals.get('sh/oz29krqh').text.split('\n');
const newBullets=[
  'Majority baseline: predicts neutral for every sentence, showing the effect of class imbalance.',
  'Trained baseline: the 726 test rows were excluded from vocabulary fitting, training and model selection.',
  'FinBERT: original training included PhraseBank, so this is a diagnostic rather than a fair unseen-data comparison.',
];
for(let i=2;i>=0;i--){
  const oldColon=oldBullets[i].indexOf(':'),newColon=newBullets[i].indexOf(':');
  interpretation.text.replace(oldBullets[i].slice(oldColon+2),newBullets[i].slice(newColon+2));
  interpretation.text.replace(oldBullets[i].slice(0,oldColon+1),newBullets[i].slice(0,newColon+1));
}

text('sh/dgbulwnm','Model Choice and Training Exposure');
table('tb/8bm5cnad',[
  ['Aspect','Trained TF-IDF + LR','Pretrained FinBERT'],
  ['Representation','Word-frequency n-grams','Contextual BERT features'],
  ['Training here','Fitted on saved training split','No project fine-tuning'],
  ['PhraseBank exposure','Only our training partition','Checkpoint trained on PhraseBank'],
  ['Evaluation role','Held-out test measurement','Diagnostic with prior exposure'],
  ['Known limitations','Limited context representation','Overlap limits generalization claims'],
]);
labeled('sh/yhkbe1o7','Key observation: ','a higher diagnostic score does not prove better generalization when training histories differ. New human-reviewed examples are needed for an independent comparison.');

paras('sh/87ipkzal',[
  'Concepts demonstrated','Transformer encoder architecture and self-attention',
  'WordPiece tokenization and pretrained inference','Supervised baseline training and validation',
  'Accuracy, macro F1 and confusion matrices','Streamlit integration for text and CSV analysis',
]);
paras('sh/ml07i9sv',['Conclusion',`The app applies pretrained FinBERT and a student-trained baseline. The baseline reaches ${pct(metrics.baseline.accuracy)} held-out accuracy. FinBERT's ${pct(metrics.finbert.accuracy)} PhraseBank score is a prior-exposure diagnostic.`]);
paras('sh/oryp8fah',['Future scope','Complete human review of 30 new candidate sentences. Explore mixed sentiment and score calibration.\nLive demo']);
resolve('sh/oryp8fah').text.get('Live demo').link={uri:live,isExternal:true};
const notes=[
  `Project: Financial News Sentiment Analysis Using FinBERT. Akshat Agarwal, 202401100100028, CS-5A. Faculty: Kriti Sharma. KIET Group of Institutions, Ghaziabad. Live app: ${live} GitHub: ${repo}`,
  'FinBERT uses BERT-base with a sequence-classification head. This project does not fine-tune FinBERT. Source: https://huggingface.co/ProsusAI/finbert and its configuration.',
  'Data source: https://huggingface.co/datasets/takala/financial_phrasebank . Raw 4,846 rows; six matching duplicates removed; four conflicting-label rows quarantined. Clean 4,836 rows. Seed 42 splits: 3,385 / 725 / 726, with zero normalized overlap. C=2 selected on validation macro F1. Evidence: outputs/dataset_audit.json and models/baseline_metadata.json.',
  'All values come from outputs/evaluation.json. Majority class is selected from training labels only. The conventional baseline uses an untouched test partition. FinBERT was fine-tuned on Financial PhraseBank, so its diagnostic score is not an independent result or fair unseen-data ranking.',
  'Different training histories limit score comparisons. Independent human review of data/review/candidates.csv remains pending. No independent accuracy is claimed.',
  `Unit 4 mapping: BERT, self-attention, tokenization, inference and application integration. The conventional baseline is trained using supervised learning. Live app: ${live} GitHub: ${repo}`,
];
for(let i=0;i<6;i++)deck.slides.items[i].speakerNotes.text=notes[i];
await fs.writeFile(path.join(build,'adapted.ndjson'),(await deck.inspect({select:'$.slides[*]',include:['id','index','elements.id','elements.kind','elements.text','elements.position','elements.style']})).ndjson);
for(const [i,slide]of deck.slides.items.entries()){
  await fs.writeFile(path.join(build,`draft-${i+1}.png`),new Uint8Array(await(await deck.export({slide,format:'png',scale:1})).arrayBuffer()));
}
const candidatePath=path.join(build,`candidate-${run}.pptx`);
await(await PresentationFile.exportPptx(deck)).save(candidatePath);
const finalPath=path.join(finalDir,'financial_sentiment.pptx');
const referenceHash=crypto.createHash('sha256').update(await fs.readFile(templatePath)).digest('hex');
const result=await finalizePresentation({
  workspaceDir:ROOT,candidatePath,finalPath,pythonExecutable:path.join(RUNTIME,'python/python.exe'),
  integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','9144000,5143500','--validate-heading-fit','--validate-bullet-geometry','--require-native-table-slide','3','--require-native-table-slide','5'],
  explicitTotalSlideCount:6,requiredNativeTableOwnerSlides:[3,5],requiredNativeChartOwnerSlides:[],
  fontPolicy:{basis:'reference',families:['Calibri'],referencePath:templatePath,referenceSha256:referenceHash},
  verifyArtifactToolImport:true,receiptPath:path.join(build,`validation-${run}.json`),
});
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(const [i,slide]of finalDeck.slides.items.entries()){
  await fs.writeFile(path.join(build,`final-${i+1}.png`),new Uint8Array(await(await finalDeck.export({slide,format:'png',scale:1})).arrayBuffer()));
}
await fs.writeFile(path.join(build,'latest.json'),JSON.stringify({finalPath,referenceHash,sha256:result.finalSha256},null,2));
console.log(JSON.stringify({finalPath,sha256:result.finalSha256,integrity:result.packageIntegrity.status,layoutFindings:result.presentationLayout.findingCount}));
process.exit(0);
