"""Create a copy of English Express with a local JSON card import button.

Usage: python integrate_english_express.py C:\\effort\\pron-coach\\english-express.html
The original file is read only and is never changed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ADDON = r'''
/* English Talk local card import: injected inside the original app closure. */
(function(){
  const trigger=document.createElement('button');
  trigger.textContent='英会話カードを取り込む';
  trigger.type='button';
  trigger.style.cssText='position:fixed;right:12px;bottom:12px;z-index:9999;padding:10px 14px;border-radius:10px;background:#215a9a;color:white;border:0;cursor:pointer';
  const input=document.createElement('input');
  input.type='file';input.accept='.json,application/json';input.hidden=true;
  trigger.onclick=()=>input.click();
  input.onchange=async()=>{
    if(!input.files||!input.files[0])return;
    try{
      const cards=JSON.parse(await input.files[0].text());
      if(!Array.isArray(cards)||cards.length>500)throw Error('500枚以内のカード配列を選んでください');
      let count=0;
      for(const c of cards){
        if(!c||typeof c.en!=='string'||typeof c.ja!=='string'||!c.en.trim()||!c.ja.trim())continue;
        const topic=String(c.topic||'英会話');
        if(Object.values(S.cards).some(x=>x.en===c.en&&x.topic===topic))continue;
        put('cards',uid('u'),{en:c.en.slice(0,80),ja:c.ja.slice(0,80),
          ex:String(c.ex||'').slice(0,240),exJa:String(c.exJa||'').slice(0,240),
          topic:topic.slice(0,80),created:Date.now()+count});
        count++;
      }
      toast(count+'枚の英会話カードを追加しました');
      renderUserCards();
    }catch(e){alert('取り込みに失敗しました: '+e.message);}
    input.value='';
  };
  document.body.append(trigger,input);
})();
'''


def main():
    if len(sys.argv) != 2:
        raise SystemExit("English Express の english-express.html のパスを指定してください")
    source = Path(sys.argv[1]).resolve()
    text = source.read_text(encoding="utf-8")
    marker = "})();\n</script>\n</body></html>"
    if text.count(marker) != 1 or "const S={}" not in text or "function put(col,id,obj)" not in text:
        raise SystemExit("既存のカード保存コードと挿入位置を確認できませんでした")
    output = Path(__file__).resolve().parent / "English-Express-with-import.html"
    output.write_text(text.replace(marker, ADDON + marker), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
