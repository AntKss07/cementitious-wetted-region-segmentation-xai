// Optional browser integration checks. Install Playwright or set
// PLAYWRIGHT_MODULE to an existing playwright package directory.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({headless: true, ...(process.env.BROWSER_CHANNEL ? {channel:process.env.BROWSER_CHANNEL} : {})});
  const page = await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];
  page.on('pageerror', error => errors.push(error.message));
  const base=process.env.RESEARCH_URL || 'http://127.0.0.1:8765';
  const shots=process.env.SCREENSHOT_DIR;
  if(shots) fs.mkdirSync(shots,{recursive:true});
  const shot=async name=>{
    if(!shots)return;
    await page.locator('img').evaluateAll(images=>images.forEach(image=>image.loading='eager'));
    await page.waitForFunction(()=>[...document.querySelectorAll('img')].every(i=>i.complete));
    await page.screenshot({path:path.join(shots,name+'.png'),fullPage:true});
  };
  const navigate=async (hash,heading)=>{
    await page.goto(base+'/#'+hash);
    await page.getByRole('heading',{name:heading,exact:true}).waitFor();
    await page.locator('main[aria-busy]').waitFor({state:'detached'});
  };
  try {
    await navigate('overview','Research overview');
    await page.getByText('1,250',{exact:true}).waitFor();
    await shot('overview-desktop');
    await navigate('segmentation','Segmentation explorer');
    await page.locator('.model-panel img').first().waitFor();
    await page.waitForFunction(()=>[...document.querySelectorAll('.model-panel img')].every(i=>i.complete&&i.naturalWidth>0));
    assert.equal(await page.locator('.model-panel img').count(),4);
    await shot('segmentation-desktop');
    // Uploaded images must never inherit precomputed results from the dataset.
    await page.locator('#image-upload').setInputFiles(path.join(__dirname,'../video-frame/video-frame/frames/video 21/ezgif-frame-040.jpg'));
    await page.getByRole('heading',{name:'Uploaded image',exact:true}).waitFor();
    assert.equal(await page.locator('.model-panel img').count(),0);
    assert.equal(await page.locator('#video').count(),0);
    assert.equal(await page.getByText('Predicted WAR',{exact:true}).count(),0);
    await page.getByRole('button',{name:'Inspect uploaded image at full size'}).click();
    await page.waitForFunction(()=>document.querySelector('#dialog-image').naturalWidth>0);
    await page.keyboard.press('Escape');
    await page.locator('#image-upload').setInputFiles({name:'broken.png',mimeType:'image/png',buffer:Buffer.from('invalid image')});
    await page.getByRole('alert').filter({hasText:'could not be decoded'}).waitFor();
    assert.equal(await page.getByRole('heading',{name:'Uploaded image',exact:true}).count(),1);
    await page.locator('#image-upload').setInputFiles({name:'notes.txt',mimeType:'text/plain',buffer:Buffer.from('not an image')});
    await page.getByRole('alert').filter({hasText:'Choose a JPEG'}).waitFor();
    await page.getByRole('button',{name:'Return to dataset',exact:true}).click();
    await page.locator('#video').waitFor();
    assert.equal(await page.locator('.model-panel img').count(),4);
    await page.getByLabel('Prediction mode').selectOption('physics_enforced');
    await page.getByRole('button',{name:'Error map',exact:true}).click();
    await page.locator('.model-panel .empty').first().waitFor();
    assert.equal(await page.locator('.model-panel img').count(),3);
    assert.match(await page.locator('.model-panel').first().innerText(),/saved error map uses raw/);
    await page.getByRole('button',{name:'Predicted mask',exact:true}).click();
    await page.getByLabel('Video / specimen').selectOption('video 1');
    await page.getByLabel('Frame',{exact:true}).selectOption('ezgif-frame-001');
    await page.waitForFunction(()=>document.querySelector('.context-line')?.textContent.includes('ezgif-frame-001'));
    assert.equal(await page.locator('.model-panel img').count(),1);
    await page.getByRole('button',{name:'Inspect SegFormer-B1 prediction',exact:true}).click();
    assert.equal(await page.getByRole('dialog').isVisible(),true);
    await page.keyboard.press('Escape');
    assert.equal(await page.getByRole('dialog').isVisible(),false);
    const download = page.waitForEvent('download');
    await page.getByRole('link',{name:'Download comparison'}).click();
    assert.equal((await download).suggestedFilename(),'segmentation-comparison.png');
    await navigate('benchmark','Model benchmark');
    await page.getByLabel('Compare metric').selectOption('war_mae');
    await page.getByRole('heading',{name:'WAR MAE comparison'}).waitFor();
    await page.getByLabel('Prediction mode').selectOption('physics_enforced');
    await page.getByText('The published physics-enforced row', {exact:false}).waitFor();
    await navigate('xai','Explainability laboratory');
    await page.getByLabel('Case category').selectOption('Best Segmentation');
    await page.locator('[data-case-video]').first().click();
    await page.locator('.wide-figure img').waitFor();
    await shot('xai-desktop');
    await navigate('dataset','Dataset & annotations');
    await page.getByLabel('Dataset split').selectOption('train');
    await page.getByLabel('Video / specimen').selectOption('video 2');
    await page.waitForFunction(()=>document.querySelector('.context-line')?.textContent.includes('video 2'));
    assert.equal(await page.locator('.four-columns img').count(),4);
    await page.waitForFunction(()=>[...document.querySelectorAll('img')].every(i=>i.complete&&i.naturalWidth>0));
    await navigate('outputs','Research outputs');
    await page.getByLabel('Search artifacts').fill('metrics_test');
    assert.equal(await page.locator('#artifact-results tbody tr').count(),5);
    await page.getByLabel('File type').selectOption('PDF');
    await page.getByText('No artifacts match these filters.',{exact:true}).waitFor();
    await navigate('vlm','VLM analysis');
    await page.getByText('PLANNED · UNAVAILABLE',{exact:true}).waitFor();
    for(const width of [820,390]) {
      await page.setViewportSize({width,height:1000});
      await navigate('segmentation','Segmentation explorer');
      await page.waitForFunction(()=>[...document.querySelectorAll('img')].every(i=>i.complete&&i.naturalWidth>0));
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`Horizontal overflow at ${width}`);
      await shot(`segmentation-${width}`);
    }
    assert.deepEqual(errors,[]);
    console.log('PASS: navigation, frame synchronization, modes, missing artifacts, image dialog, downloads, benchmark controls, XAI cases, annotations, search, planned VLM and responsive layouts.');
  } catch(error) {
    await shot('failure');
    console.error(await page.locator('main').innerText());
    throw error;
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
