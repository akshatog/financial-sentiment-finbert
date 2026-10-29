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
const evaluation=JSON.parse(await fs.readFile(path.join(ROOT,'outputs/finetuning/evaluation.json'),'utf8'));
const manifest=JSON.parse(await fs.readFile(path.join(ROOT,'models/finbert_finetuned/training_manifest.json'),'utf8'));
const history=JSON.parse(await fs.readFile(path.join(ROOT,'models/finbert_finetuned/training_history.json'),'utf8'));
if(manifest.status!=='completed'||manifest.method!=='full_fine_tuning')throw new Error('Completed full fine-tuning evidence is required.');
if(evaluation.test_fingerprint!==metrics.test_fingerprint)throw new Error('The reported models must share the same test split.');
const tuned=evaluation.metrics;
// A final evaluation of the restored checkpoint repeats epoch 3 in Trainer history.
// Keep the first evaluation at each integer epoch to show the actual epoch trajectory.
const epochs=[...new Map(history.filter(r=>Number.isInteger(r.epoch)&&r.eval_macro_f1!==undefined).reverse().map(r=>[r.epoch,r])).values()].sort((a,b)=>a.epoch-b.epoch);
if(epochs.length!==3)throw new Error('Expected three per-epoch validation results.');
const selected=epochs.reduce((best,row)=>row.eval_macro_f1>best.eval_macro_f1?row:best);
const deck=await PresentationFile.importPptx(await FileBlob.load(templatePath));
if(deck.slides.items.length!==6)throw new Error('Expected the inspected six-slide reference.');
// Refresh the reference renders from the actual input before editing it.
for(const [i,slide]of deck.slides.items.entries()){
  await fs.writeFile(path.join(build,`reference-${i+1}.png`),new Uint8Array(await(await deck.export({slide,format:'png',scale:1})).arrayBuffer()));
}
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
labeled('sh/sryl4zqx','Model: ','HKUST FinBERT with full supervised fine-tuning on the project training split.');
labeled('sh/utg3698n','Stack: ','Python, Transformers, PyTorch, scikit-learn and Streamlit.');
labeled('sh/hofulsf2','Objective: ','Compare a fine-tuned Transformer against majority and TF-IDF baselines on the same test split.');
const identity=deck.slides.items[0].shapes.add({geometry:'textbox',position:{left:57.6,top:340.8,width:403.2,height:134.4},fill:'none',line:{fill:'none',width:0}});
identity.text='Akshat Agarwal\nRoll No. 202401100100028, CS-5A\nFaculty: Kriti Sharma\nKIET Group of Institutions, Ghaziabad';
identity.text.style={typeface:'Calibri',fontSize:17.33,color:'#FFFFFF'};identity.text.insets=0;

// Reuse all five native flow nodes and explanation panels.
text('sh/b29kza94','Financial Text');text('sh/ove9o7yd','BERT Encoder');
text('sh/nu58f2hs','Classification Head');text('sh/ahwrqhgj','Sentiment Label');
paras('sh/cnupgny5',['Tokenization','Converts financial text into WordPiece token IDs.']);
paras('sh/t8byxkn2',['Encoder','Uses self-attention to represent words in their surrounding context.']);
paras('sh/76twva5c',['Classification head','Maps the encoded statement to three sentiment logits.']);
paras('sh/hcvy14ne',['Full fine-tuning','Updates all BERT weights and the classification head using labeled training sentences.']);
paras('sh/fadgzu58',['Inference','Softmax produces three model scores. The highest named score selects the sentiment label.']);

text('sh/cza94vmx','Training Pipeline: Three Model Routes');
table('tb/juho7qhg',[
  ['Setting','Majority baseline','TF-IDF + logistic regression','Student fine-tuned FinBERT'],
  ['Representation','Class frequency','Word unigrams + bigrams','Contextual BERT features'],
  ['Training here','Training-label counts','3,385 training rows','3,385 rows, full fine-tuning'],
  ['Selection','Neutral majority','C = 2 on validation macro F1','Epoch 2 on validation macro F1'],
  ['Main purpose','Simple reference','Conventional baseline','Main application model'],
]);
paras('sh/3ah8rqlg',['Fine-tuning configuration','3 epochs, learning rate 2e-5, max length 128. Effective batch size 16 with FP16 on a Tesla T4.']);
paras('sh/pcjqtg36',['Checkpoint selection','725 validation rows select checkpoint 424 at epoch 2. The test split stays outside training and selection.']);
text('sh/m5cra54z',`PhraseBank: ${audit.main_dataset.clean_rows.toLocaleString('en-US')} clean rows. Train / validation / test: 3,385 / 725 / 726. Seed 42.`);

// Keep the result cards and use the actual classification metrics.
text('sh/rm1k7yt4','Majority baseline');
paras('sh/ql8jytsj',['Accuracy',pct(metrics.majority_baseline.accuracy)]);
paras('sh/doj29oba',['Macro F1',pct(metrics.majority_baseline.macro_f1)]);
text('sh/3ihk3et8','Simple reference');
text('sh/kbm987y5','Trained TF-IDF + LR');
paras('sh/5cva1cfq',['Accuracy',pct(metrics.baseline.accuracy)]);
paras('sh/i94r6xgz',['Macro F1',pct(metrics.baseline.macro_f1)]);
text('sh/w72947yt','Conventional baseline');
text('sh/a5kr2xg3','Fine-tuned FinBERT');
paras('sh/v6tsv2xo',['Accuracy',pct(tuned.accuracy)]);
paras('sh/g36tgryd',['Macro F1',pct(tuned.macro_f1)]);
text('sh/p0batw72','Selected checkpoint');
resolve('sh/ih8ju9sn').fill='#EAF4F4';
resolve('sh/x8vaxsfe').fill='#028090';
for(const id of ['sh/kbm987y5','sh/5cva1cfq','sh/i94r6xgz'])resolve(id).text.style={color:'#0B3C49'};
for(const id of ['sh/a5kr2xg3','sh/v6tsv2xo','sh/g36tgryd'])resolve(id).text.style={color:'#FFFFFF'};
for(const el of records[3].elements.filter(el=>el.kind==='shape')){
  if(Math.abs(el.position.top-273.6)<.1&&Math.abs(el.position.left-67.2)<.1)resolve(el.id).fill='#FFFFFF';
  if(Math.abs(el.position.top-273.6)<.1&&Math.abs(el.position.left-662.4)<.1)resolve(el.id).fill='#FFFFFF';
}
const interpretation=resolve('sh/oz29krqh');
const oldBullets=originals.get('sh/oz29krqh').text.split('\n');
const newBullets=[
  'Test split: 726 sentences, excluded from training and checkpoint selection.',
  `Fine-tuned gain: +${(evaluation.same_test_baseline_comparison.accuracy_change*100).toFixed(2)} percentage points accuracy and +${(evaluation.same_test_baseline_comparison.macro_f1_change*100).toFixed(2)} points macro F1 over TF-IDF.`,
  `Separate reference: Prosus scores ${pct(metrics.finbert.accuracy)} / ${pct(metrics.finbert.macro_f1)} with prior PhraseBank training. This is not a fair benchmark.`,
];
for(let i=2;i>=0;i--){
  const oldColon=oldBullets[i].indexOf(':'),newColon=newBullets[i].indexOf(':');
  interpretation.text.replace(oldBullets[i].slice(oldColon+2),newBullets[i].slice(newColon+2));
  interpretation.text.replace(oldBullets[i].slice(0,oldColon+1),newBullets[i].slice(0,newColon+1));
}

text('sh/dgbulwnm','Fine-Tuning and Checkpoint Selection');
table('tb/8bm5cnad',[
  ['Stage / setting','Recorded value','Evidence / outcome'],
  ...epochs.map(row=>[`Epoch ${row.epoch}${row.epoch===selected.epoch?' (selected)':''}`,`${pct(row.eval_macro_f1)} validation macro F1`,`Optimizer step ${row.step}`]),
  ['Full fine-tuning',`${(manifest.parameter_counts.trainable/1e6).toFixed(2)}M trainable parameters`,'BERT encoder and classifier updated'],
  ['Compute','Tesla T4, about 7 min 50 sec',`${manifest.global_optimizer_steps} optimizer steps across 3 epochs`],
]);
labeled('sh/yhkbe1o7','Key observation: ',`validation macro F1 peaks at epoch ${selected.epoch}. We restore checkpoint ${selected.step} before testing. Results describe one seeded split, so broader generalization needs independent examples.`);

paras('sh/87ipkzal',[
  'Concepts demonstrated','Transformer encoder architecture and self-attention',
  'WordPiece tokenization and input preparation','Full supervised fine-tuning of a pretrained BERT',
  'Accuracy, macro F1 and confusion matrices','Streamlit integration for text and CSV analysis',
]);
paras('sh/ml07i9sv',['Conclusion',`Fine-tuned FinBERT reaches ${pct(tuned.accuracy)} accuracy and ${pct(tuned.macro_f1)} macro F1 on 726 test sentences. It improves accuracy by ${(evaluation.same_test_baseline_comparison.accuracy_change*100).toFixed(2)} percentage points over TF-IDF on the same split.`]);
paras('sh/oryp8fah',['Future scope','Evaluate new human-reviewed sentences. Study mixed sentiment and score calibration.\nLive demo']);
resolve('sh/oryp8fah').text.get('Live demo').link={uri:live,isExternal:true};
const notes=[
  `Project: Financial News Sentiment Analysis Using FinBERT. Akshat Agarwal, 202401100100028, CS-5A. Faculty: Kriti Sharma. KIET Group of Institutions, Ghaziabad. Live app: ${live} GitHub: ${repo}`,
  'The project uses yiyanghkust/finbert-pretrain, a finance-domain pretrained BERT model, and fully fine-tunes its encoder and a three-class classification head on the saved training split. All 109,754,115 parameters are trainable. An attention-query weight changes after training. Source: https://huggingface.co/yiyanghkust/finbert-pretrain . Evidence: models/finbert_finetuned/training_manifest.json. This is distinct from the published ProsusAI/finbert sentiment checkpoint.',
  'Data source: https://huggingface.co/datasets/takala/financial_phrasebank . Raw 4,846 rows; six matching duplicates removed; four conflicting-label rows quarantined. Clean 4,836 rows. Seed 42 splits: 3,385 / 725 / 726, with zero normalized overlap. C=2 selected on validation macro F1. Evidence: outputs/dataset_audit.json and models/baseline_metadata.json.',
  `Metrics come from outputs/evaluation.json and outputs/finetuning/evaluation.json. Majority, TF-IDF and student fine-tuned FinBERT use the same 726-row test split. Student model accuracy: ${tuned.accuracy}; macro F1: ${tuned.macro_f1}. Gains over TF-IDF are descriptive percentage-point differences, not a statistical significance claim. Separate Prosus reference has known PhraseBank sentiment-training exposure and is not a fair unseen-data benchmark. Unpublished pretraining overlap for HKUST is not established.`,
  `Evidence: models/finbert_finetuned/training_history.json and training_manifest.json. Per-epoch validation macro F1: ${epochs.map(r=>`epoch ${r.epoch} ${pct(r.eval_macro_f1)} at step ${r.step}`).join(', ')}. Best checkpoint: ${manifest.selected_checkpoint}. The final history entry re-evaluates restored checkpoint 424 and must not replace the original epoch-3 result. Run elapsed time ${manifest.elapsed_seconds} seconds, 636 optimizer steps, Tesla T4. Independent human review of new examples remains pending.`,
  `Unit 4 mapping: BERT encoder architecture, self-attention, WordPiece tokenization, the data-to-fine-tuning pipeline and applied text sentiment analysis. Full supervised fine-tuning updates all BERT parameters. Live app: ${live} GitHub: ${repo}`,
];
for(let i=0;i<6;i++)deck.slides.items[i].speakerNotes.text=notes[i];
await fs.writeFile(path.join(build,'adapted.ndjson'),(await deck.inspect({select:'$.slides[*]',include:['id','index','elements.id','elements.kind','elements.text','elements.position','elements.style']})).ndjson);
for(const [i,slide]of deck.slides.items.entries()){
  await fs.writeFile(path.join(build,`draft-${i+1}.png`),new Uint8Array(await(await deck.export({slide,format:'png',scale:1})).arrayBuffer()));
}
const candidatePath=path.join(build,`candidate-${run}.pptx`);
await(await PresentationFile.exportPptx(deck)).save(candidatePath);
const finalPath=path.join(finalDir,'financial_sentiment_finetuned.pptx');
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
let deliveredPath=path.join(ROOT,'presentation/financial_sentiment_revised.pptx');
try{await fs.copyFile(finalPath,deliveredPath);}
catch(error){
  if(!['EBUSY','EACCES','EPERM'].includes(error.code))throw error;
  deliveredPath=path.join(ROOT,'presentation/financial_sentiment_finetuned.pptx');
  await fs.copyFile(finalPath,deliveredPath);
}
const deliveredHash=crypto.createHash('sha256').update(await fs.readFile(deliveredPath)).digest('hex');
if(deliveredHash!==result.finalSha256)throw new Error('Delivered file differs from the validated output.');
console.log(JSON.stringify({finalPath:deliveredPath,validatedPath:finalPath,sha256:result.finalSha256,integrity:result.packageIntegrity.status,layoutFindings:result.presentationLayout.findingCount,metrics:{accuracy:tuned.accuracy,macroF1:tuned.macro_f1},selectedEpoch:selected.epoch,selectedStep:selected.step}));
process.exit(0);
