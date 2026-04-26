/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useRef, useMemo, Fragment } from "react";
import { GoogleGenAI, Type } from "@google/genai";
import { fastMap2D, cosineSimilarity } from "./lib/math";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const playChime = () => {
    try {
        const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
        const ctx = new AudioContext();
        
        const playTone = (freq: number, startTime: number, duration: number) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, startTime);
            gain.gain.setValueAtTime(0.5, startTime);
            gain.gain.exponentialRampToValueAtTime(0.01, startTime + duration);
            osc.start(startTime);
            osc.stop(startTime + duration);
        };

        const now = ctx.currentTime;
        playTone(523.25, now, 0.3); // C5
        playTone(659.25, now + 0.15, 0.4); // E5
    } catch(e) {}
};

async function withRetry<T>(fn: () => Promise<T>, apiName: string, maxRetries = 3, setStatus?: (msg: string) => void): Promise<T> {
  let retries = 0;
  while (true) {
    try {
      return await fn();
    } catch (e: any) {
      const errorString = typeof e === 'object' ? JSON.stringify(e) + String(e.message) : String(e);
      if ((errorString.includes("429") || errorString.includes("RESOURCE_EXHAUSTED") || errorString.includes("quota")) && retries < maxRetries) {
        retries++;
        // Wait longer for each retry (5, 10, 20s) to give the free tier quota time to reset. RPM limit is usually 15.
        const delayMs = 5000 * Math.pow(2, retries - 1);
        const msg = `Rate limit hit on ${apiName}, retrying in ${delayMs/1000}s... (${retries}/${maxRetries})`;
        console.warn(msg);
        if (setStatus) setStatus(msg);
        await new Promise(resolve => setTimeout(resolve, delayMs));
      } else {
        throw e;
      }
    }
  }
}

export default function App() {
  const [processed, setProcessed] = useState<any[]>([]);
  const [selectedPoint, setSelectedPoint] = useState<any>(null);
  const [showImageOverlay, setShowImageOverlay] = useState(false);
  const [loading, setLoading] = useState(false);
  const [enableFraudDetection, setEnableFraudDetection] = useState(true);
  const [status, setStatus] = useState("");
  const [cost1k, setCost1k] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const processFiles = async (newFiles: File[]) => {
    setLoading(true);
    setCost1k(null);
    let totalInputTokens = 0;
    let totalOutputTokens = 0;
    
    try {
      const results: any[] = [];
      for (let i = 0; i < newFiles.length; i++) {
        const file = newFiles[i];
        setStatus(`Analyzing Receipt ${i + 1} of ${newFiles.length}...`);
        const dataUrl = await new Promise<string>((res) => {
          const reader = new FileReader();
          reader.onload = () => res(reader.result as string);
          reader.readAsDataURL(file);
        });
        const base64Data = dataUrl.split(',')[1];
        const mimeType = file.type;

        try {
          // 1. Analyze Receipt properties
          let analysisObj: any = {
            merchant: "Unknown",
            date: "Unknown",
            totalAmount: 0.0,
            anomalies: [],
            fraudScore: 0,
            isFraudulent: false,
            fraudReason: enableFraudDetection ? "" : "Fraud detection disabled",
            receiptText: ""
          };

          const currentDate = new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
          const extractionPrompt = enableFraudDetection 
            ? `You are an expert fraud investigator analyzing Dutch receipts. Today's exact date is: ${currentDate}. Any receipt date before today is in the past and is NOT a future date. Carefully examine the image for signs of financial fraud, tampering, or copy-paste forgery (e.g., duplicated receipts with altered values). Check for: 1) Math errors, 2) Font/typography inconsistencies, 3) Suspicious items/amounts. DO NOT flag based on future dates or timestamps, completely ignore dates/times when evaluating fraud. Output JSON extracting: merchant name, totalAmount, date, a list of specific 'anomalies', a 'fraudScore' (0-100), a boolean 'isFraudulent' (true if score > 50), and a 'fraudReason' (Start with a VERY brief 1-sentence concise explanation, then provide full details below it). Also extract a raw ascii text representation of the receipt contents ('receiptText'). ALWAYS output valid JSON matching the exact schema provided.`
            : "Extract data from this receipt. Focus only on factual extraction. Output JSON extracting: merchant name, totalAmount, date, and a raw ascii text representation of the receipt contents ('receiptText'). Fill in the other required schema fields with default empty values: anomalies: [], fraudScore: 0, isFraudulent: false, fraudReason: 'Disabled'. ALWAYS output valid JSON matching the exact schema provided.";

          const analysisResponse = await withRetry(() => ai.models.generateContent({
            model: "gemini-flash-latest",
            contents: {
              parts: [
                { text: extractionPrompt },
                { inlineData: { data: base64Data, mimeType } }
              ]
            },
            config: {
              temperature: 0.0,
              responseMimeType: "application/json",
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  merchant: { type: Type.STRING },
                  date: { type: Type.STRING },
                  totalAmount: { type: Type.NUMBER },
                  anomalies: { type: Type.ARRAY, items: { type: Type.STRING } },
                  fraudScore: { type: Type.NUMBER, description: "Score from 0 to 100 indicating likelihood of fraud" },
                  isFraudulent: { type: Type.BOOLEAN },
                  fraudReason: { type: Type.STRING },
                  receiptText: { type: Type.STRING, description: "A raw ascii text representation of the receipt contents" }
                }
              }
            }
          }), 'generateContent (Analysis)', 3, setStatus);
          
          if (analysisResponse.usageMetadata) {
            totalInputTokens += analysisResponse.usageMetadata.promptTokenCount || 0;
            totalOutputTokens += analysisResponse.usageMetadata.candidatesTokenCount || 0;
          }

          try { 
            const parsed = JSON.parse(analysisResponse.text || "{}");
            analysisObj = { ...analysisObj, ...parsed };
          } catch (e) { 
            console.error(e); 
          }

          // 2. Fact-Based Vector Embedding for Duplicate Detection
          const duplicateSignature = [
            `Merchant: ${analysisObj.merchant}`,
            `Date/Time: ${analysisObj.date}`,
            `Total: ${analysisObj.totalAmount}`,
            `Receipt Output: ${analysisObj.receiptText}`
          ].join("\n");

          const embedResponse = await withRetry(() => ai.models.embedContent({
            model: 'gemini-embedding-2-preview',
            contents: duplicateSignature
          }), 'embedContent (Similarity)', 3, setStatus);
          
          results.push({
            id: file.name + Date.now() + Math.random(),
            file,
            dataUrl,
            analysis: analysisObj,
            embedding: embedResponse.embeddings?.[0]?.values || []
          });

          // Pacing to avoid rate limit (RESOURCE_EXHAUSTED 429)
          if (i < newFiles.length - 1) {
            setStatus("Pacing API requests to avoid rate limits...");
            await new Promise(resolve => setTimeout(resolve, 4000));
          }

        } catch (e: any) {
          console.error("API Error details:", e);
          const errorString = typeof e === 'object' ? JSON.stringify(e) + String(e.message) : String(e);
          const isQuota = errorString.includes("429") || errorString.includes("quota") || errorString.includes("RESOURCE_EXHAUSTED");
          
          if (isQuota) {
            alert(`Gemini API Quota Exceeded!\n\nYou have reached the Gemini API daily limits or rate limits.\nSuccessfully processed ${results.length} items before failing.\n\nPlease check your AI Studio plan and billing details.`);
            break; // Break the upload loop, but continue rendering what we have
          } else {
            alert(`Failed to process ${file.name}: ${e?.message || 'Unknown API Error'}`);
            break; // Break loop on other errors too
          }
        }
      }

      if (results.length === 0) {
         setStatus("No items were processed due to API limits or errors.");
         setLoading(false);
         return;
      }

      setStatus("Mapping Vectors...");
      const currentProcessed = [...processed, ...results];
      const validProcessed = currentProcessed.filter(p => p.embedding && p.embedding.length > 0);
      
      const embeddings = validProcessed.map(r => r.embedding);
      const coords = fastMap2D(embeddings);
      
      // Keep track of overlap to slightly jitter identical points visually
      const overlapCounts: Record<string, number> = {};
      
      const finalData = validProcessed.map((r, i) => {
        let x = coords[i].x;
        let y = coords[i].y;
        
        // Jitter handling for exact overlaps
        const key = `${Math.round(x * 100)},${Math.round(y * 100)}`;
        if (overlapCounts[key] !== undefined) {
           overlapCounts[key]++;
           // deterministic spiral offset for overlaps
           const count = overlapCounts[key];
           const angle = count * Math.PI / 4;
           // Roughly 3% distance jitter per count factor
           const radius = 0.02 + (count * 0.015);
           x += Math.cos(angle) * radius;
           y += Math.sin(angle) * radius;
        } else {
           overlapCounts[key] = 0;
        }

        return {
          ...r,
          x,
          y
        };
      });

      setProcessed(finalData);

      if (finalData.length > 0 && !selectedPoint) {
        setSelectedPoint(finalData[0]);
      }

      // Calculate cost
      if (results.length > 0) {
        const flashInputCost = 0.075; // USD/1M tokens
        const flashOutputCost = 0.30;
        // ~ 0.02 embedding cost, negligible, just estimate flash
        const avgInput = totalInputTokens / results.length;
        const avgOutput = totalOutputTokens / results.length;
        const estimated1kCost = ((avgInput * 1000) / 1000000) * flashInputCost + ((avgOutput * 1000) / 1000000) * flashOutputCost;
        // Assume roughly 1 USD = 0.95 EUR, or just treat as EUR. We'll use 0.95 factor:
        setCost1k(estimated1kCost * 0.95);
      }

      playChime();
    } catch (e: any) {
      console.error(e);
      alert("System Error: " + (e.message || "Unknown error occurred"));
    } finally {
      setLoading(false);
      setStatus("");
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files).filter((f: any) => f.type.startsWith('image/'));
    if (droppedFiles.length) processFiles(droppedFiles as File[]);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files).filter((f: any) => f.type.startsWith('image/'));
      if (selectedFiles.length) processFiles(selectedFiles as File[]);
    }
  };

  const anomalousCount = processed.filter(p => p.analysis?.isFraudulent).length;

  const duplicateEdges = useMemo(() => {
    const pairs: { type: 'exact' | 'near', p1: any, p2: any }[] = [];
    for (let i = 0; i < processed.length; i++) {
        for (let j = i + 1; j < processed.length; j++) {
            // We use cosine similarity here.
            // > 0.98 indicates exact clones, > 0.90 for almost identical / near duplicates
            const sim = cosineSimilarity(processed[i].embedding, processed[j].embedding);
            if (sim > 0.98) { 
                pairs.push({ type: 'exact', p1: processed[i], p2: processed[j] });
            } else if (sim > 0.90) {
                pairs.push({ type: 'near', p1: processed[i], p2: processed[j] });
            }
        }
    }
    return pairs;
  }, [processed]);

  const getConnectedReceipts = (point: any) => {
    if (!point) return [];
    const connected = new Set<any>();
    connected.add(point);
    duplicateEdges.forEach(e => {
        if (e.p1.id === point.id) connected.add(e.p2);
        if (e.p2.id === point.id) connected.add(e.p1);
    });
    return Array.from(connected);
  };

  return (
    <div 
      className="bg-[#F5F5F5] w-full min-h-screen overflow-hidden flex flex-col font-sans text-[#1A1A1A]"
      onDragOver={e => e.preventDefault()}
      onDrop={handleDrop}
    >
      <header className="h-20 border-b border-[#000] flex items-center justify-between px-8 bg-white shrink-0">
        <div className="flex items-baseline gap-4">
          <h1 className="text-4xl font-black tracking-tighter leading-none uppercase">
            Kwitti. <span className="text-sm font-normal tracking-normal normal-case opacity-50 hidden sm:inline">Receipt Vector Encoder v1.02</span>
          </h1>
        </div>
        <div className="flex gap-6 items-center uppercase text-[10px] font-bold tracking-widest">
          {loading ? (
             <div className="flex items-center gap-2 animate-pulse">
               <span className="w-2 h-2 rounded-full bg-orange-600"></span>
               <span>{status}</span>
             </div>
          ) : (
            <>
              <div className="hidden sm:flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-red-600"></span>
                <span>{duplicateEdges.filter(e => e.type === 'exact').length} Exact Duplicates</span>
              </div>
              <div className="hidden sm:flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-orange-400"></span>
                <span>{duplicateEdges.filter(e => e.type === 'near').length} Near Duplicates</span>
              </div>
              {enableFraudDetection && (
                <div className="hidden sm:flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-orange-600"></span>
                  <span>{anomalousCount} Frauds Detected</span>
                </div>
              )}
              <div className="hidden sm:flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-black"></span>
                <span>{processed.length} Encoded Items</span>
              </div>
            </>
          )}
          
          <label className="flex items-center gap-2 cursor-pointer border-l border-black pl-6 ml-2 transition-opacity hover:opacity-75">
            <input 
              type="checkbox" 
              checked={enableFraudDetection}
              onChange={(e) => setEnableFraudDetection(e.target.checked)}
              disabled={loading}
              className="accent-black w-3 h-3 cursor-pointer"
            />
            <span className={!enableFraudDetection ? "opacity-50" : ""}>Fraud Detection</span>
          </label>

          <input 
            type="file" 
            multiple 
            accept="image/*" 
            className="hidden" 
            ref={fileInputRef} 
            onChange={handleFileInput}
          />
          <button 
            disabled={loading}
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-black px-4 py-2 hover:bg-black hover:text-white transition-colors disabled:opacity-50 ml-2"
          >
            Upload New Batch
          </button>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        <aside className="w-16 border-r border-black hidden md:flex flex-col items-center py-8 gap-12 bg-white shrink-0">
          <div className="[writing-mode:vertical-rl] rotate-180 font-black text-xl uppercase tracking-tighter">Overview</div>
        </aside>

        <main className="flex-1 relative bg-[#EEE] p-4 flex">
          <div className="absolute top-8 left-8 z-10 bg-white/80 backdrop-blur border border-black p-4 w-64 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
            <h2 className="text-xs font-black uppercase mb-1 tracking-wider">Vector Space Map</h2>
            <p className="text-[10px] leading-tight text-gray-600">Projection: FastMap / Dimension: 768-d to 2-d<br/>Multimodal Embeddings Model</p>
            {processed.length === 0 && !loading && (
               <div className="mt-4 pt-4 border-t border-black text-[10px] font-bold text-orange-600 animate-pulse uppercase">
                 Drop receipt images anywhere to begin encoding
               </div>
            )}
          </div>

          <div className="flex-1 h-full border-2 border-black bg-white relative">
            {processed.length > 0 && (
              <svg className="w-full h-full absolute inset-0">
                {/* Draw Duplicate Relationships First (bottom-most layer) */}
                {duplicateEdges.map((edge, i) => (
                  <line 
                    key={`duplicate-${i}`}
                    x1={`${edge.p1.x * 100}%`}
                    y1={`${edge.p1.y * 100}%`}
                    x2={`${edge.p2.x * 100}%`}
                    y2={`${edge.p2.y * 100}%`}
                    stroke={edge.type === 'exact' ? "red" : "darkorange"}
                    strokeWidth={edge.type === 'exact' ? "3" : "2"}
                    strokeDasharray={edge.type === 'exact' ? "4 2" : "2 4"}
                  />
                ))}
              </svg>
            )}

            {/* Draw Items */}
            {processed.map((point) => {
              const isSelected = selectedPoint?.id === point.id;
              const isAnomalous = point.analysis?.isFraudulent;
              const isFraudDisabled = point.analysis?.fraudReason === "Fraud detection disabled";
              
              let cColor = "#000000";
              if (!isFraudDisabled && isAnomalous) cColor = "#FF4B00";
              else if (isFraudDisabled) cColor = "#6b7280"; // Gray if skipped

              return (
                <div
                  key={point.id}
                  className={`absolute -translate-x-1/2 -translate-y-1/2 rounded-full cursor-pointer transition-transform ${isSelected ? 'scale-150 z-20' : 'hover:scale-125 z-10'}`}
                  style={{ 
                    left: `${point.x * 100}%`, 
                    top: `${point.y * 100}%`,
                    width: isAnomalous ? '16px' : '10px',
                    height: isAnomalous ? '16px' : '10px',
                    backgroundColor: cColor,
                    border: `${isAnomalous ? '2px' : '1px'} solid #000`,
                    boxShadow: isSelected ? '0 4px 6px -1px rgba(0, 0, 0, 0.5)' : 'none',
                    zIndex: isSelected ? 10 : 1
                  }}
                  title={`${point.analysis?.merchant || 'Receipt'} - €${point.analysis?.totalAmount}`}
                  onClick={() => {
                    setSelectedPoint(point);
                    setShowImageOverlay(true);
                  }}
                />
              )
            })}
          </div>
        </main>

        <aside className="w-80 border-l border-black bg-white flex flex-col shrink-0 overflow-y-auto">
          {selectedPoint ? (
            <>
              <div className="p-6 border-b border-black">
                <h3 className="text-xs font-black uppercase tracking-widest mb-4 flex justify-between">
                  <span>Inspection Panel</span>
                  <span className={selectedPoint.analysis?.isFraudulent ? "text-orange-600" : ""}>
                    {selectedPoint.analysis?.isFraudulent ? "[FRAUD]" : "[OK]"}
                  </span>
                </h3>
                
                <div className="bg-gray-100 border border-black p-2 mb-4 relative group">
                  <div className="hidden group-hover:block absolute inset-0 z-10 bg-white border border-black p-1">
                      <img src={selectedPoint.dataUrl} className="w-full h-full object-contain" alt="Original Receipt" />
                  </div>
                  <div className="bg-white aspect-[3/4] flex overflow-y-auto p-4 shadow-inner">
                    <pre className="w-full text-left text-[8px] font-mono leading-[1.2] whitespace-pre-wrap">
                      {selectedPoint.analysis?.receiptText || "TEXT EXTRACTION UNAVAILABLE"}
                    </pre>
                  </div>
                  <div className="text-center text-[8px] uppercase font-bold mt-1 opacity-50">Hover to view original</div>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="block text-[9px] font-bold uppercase opacity-50">Metadata</label>
                    <p className="text-sm font-bold truncate" title={selectedPoint.analysis?.merchant}>
                      {selectedPoint.analysis?.merchant?.toUpperCase() || 'UNKNOWN MERCHANT'}
                    </p>
                    <p className="text-xs font-mono mt-1">€ {selectedPoint.analysis?.totalAmount} | {selectedPoint.analysis?.date || 'No Date'}</p>
                  </div>
                  <div>
                    <label className="block text-[9px] font-bold uppercase opacity-50">Embedding Vector (Slice 0-7)</label>
                    <div className="grid grid-cols-4 gap-1 mt-1 font-mono text-[10px]">
                      {selectedPoint.embedding.slice(0, 8).map((v: number, i: number) => (
                        <span key={i} className="bg-black text-white px-1 truncate" title={v.toString()}>
                          {v.toFixed(3)}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="flex-1 p-6">
                <h3 className="text-xs font-black uppercase tracking-widest mb-2">Analysis Result</h3>
                
                {selectedPoint.analysis?.fraudReason === "Fraud detection disabled" ? (
                  <div className="mb-4 p-2 bg-gray-100 border border-black flex justify-between items-start opacity-70">
                    <p className="text-[11px] leading-relaxed">Fraud detection was disabled during encode.</p>
                  </div>
                ) : selectedPoint.analysis?.isFraudulent ? (
                  <div className="mb-4 p-2 bg-orange-100 border border-orange-600 text-orange-900">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-bold text-orange-600 block">FLAGGED FRAUDULENT</span>
                      <span className="font-bold text-orange-800 text-xs">Score: {selectedPoint.analysis?.fraudScore}/100</span>
                    </div>
                    <p className="text-[11px] leading-relaxed">
                      {selectedPoint.analysis?.fraudReason}
                    </p>
                  </div>
                ) : (
                  <div className="mb-4 p-2 bg-gray-100 border border-black flex justify-between items-start">
                    <p className="text-[11px] leading-relaxed">Analysis passed. No immediate signs of fraud.</p>
                    <span className="font-bold text-gray-500 text-[10px]">Score: {selectedPoint.analysis?.fraudScore || 0}/100</span>
                  </div>
                )}

                {selectedPoint.analysis?.anomalies?.length > 0 && (
                   <div className="mb-4">
                     <label className="block text-[9px] font-bold uppercase opacity-50 mb-1">Detected Artifacts</label>
                     <ul className="text-[10px] list-disc pl-4 space-y-1">
                        {selectedPoint.analysis.anomalies.map((a: string, i: number) => (
                          <li key={i}>{a}</li>
                        ))}
                     </ul>
                   </div>
                )}
                
                <div className="mt-auto flex gap-2">
                  <button className={`flex-1 text-white text-[10px] font-black uppercase py-2 ${selectedPoint.analysis?.isFraudulent ? 'bg-red-600 hover:bg-black' : 'bg-black hover:bg-red-600'}`}>
                    Flag Fraud
                  </button>
                  <button className="flex-1 border border-black text-[10px] font-black uppercase py-2 hover:bg-gray-100">
                    Dismiss
                  </button>
                </div>
              </div>
            </>
          ) : (
             <div className="flex-1 flex items-center justify-center p-6 text-center text-gray-500 text-xs italic">
                Select an encoded point in the vector space to inspect details.
             </div>
          )}
        </aside>
      </div>

      <footer className="h-12 border-t border-black bg-black text-white flex items-center px-8 justify-between text-[10px] font-bold uppercase tracking-widest shrink-0">
        <div className="flex gap-8">
          <span className="flex items-center gap-2">
             <div className={`w-2 h-2 rounded-full ${loading ? 'bg-orange-500 animate-pulse' : 'bg-green-500'}`}></div>
             Session: ACTIVE
             {cost1k !== null && (
               <span className="ml-4 opacity-70">| EST. API COST (1000 RECEIPTS): €{cost1k.toFixed(4)}</span>
             )}
          </span>
        </div>
        <div>
          © 2024 KWITTI ANALYTICS GROUP — AMSTERDAM
        </div>
      </footer>

      {showImageOverlay && selectedPoint && (
        <div className="fixed inset-0 bg-black/90 z-50 flex flex-col p-8">
            <div className="flex justify-between items-center mb-6 shrink-0">
               <h2 className="text-white text-2xl font-black uppercase tracking-wider">Receipt Verification</h2>
               <button onClick={() => setShowImageOverlay(false)} className="text-white text-4xl hover:text-red-500 transition-colors">&times;</button>
            </div>
            <div className="flex-1 flex gap-6 overflow-hidden justify-center overflow-x-auto w-full">
                {(() => {
                  const receipts = getConnectedReceipts(selectedPoint);
                  return receipts.map((pt, index) => {
                    const isFirst = index === 0;
                    const isLast = index === receipts.length - 1 && receipts.length > 1;
                    
                    return (
                      <Fragment key={pt.id}>
                        {isFirst && (
                          <div className="hidden lg:flex w-72 bg-white p-6 shrink-0 flex-col overflow-y-auto rounded-lg border-2 border-white shadow-xl">
                              <h3 className="font-bold uppercase text-xs mb-2 truncate text-gray-500 border-b pb-2">{pt.file?.name} - Reason</h3>
                              <p className="text-sm whitespace-pre-wrap leading-relaxed">{pt.analysis?.fraudReason || 'No anomalies detected.'}</p>
                          </div>
                        )}

                        <div className="min-w-[300px] max-w-full lg:max-w-xl flex-1 bg-white flex flex-col h-full rounded-lg overflow-hidden border-2 border-white shadow-xl">
                            <div className="bg-gray-200 p-3 text-center text-xs font-mono font-bold border-b border-black flex justify-between">
                                <span className="truncate">{pt.file?.name || 'Receipt'}</span>
                                <span className={pt.analysis?.isFraudulent ? "text-red-600 ml-2" : "text-green-600 ml-2"}>
                                    {pt.analysis?.isFraudulent ? "[ FRAUD ]" : "[ OK ]"}
                                </span>
                            </div>
                            <div className="flex-1 overflow-auto bg-[#F5F5F5] flex justify-center items-start p-4">
                               <img src={pt.dataUrl} alt="Receipt" className="max-w-full h-auto shadow-md border border-gray-300" />
                            </div>
                            {/* Mobile/Tablet Reason View */}
                            <div className="lg:hidden bg-white p-4 border-t border-gray-300 max-h-48 overflow-y-auto text-sm">
                               <h4 className="font-bold text-xs mb-1 uppercase">Reason:</h4>
                               <p className="whitespace-pre-wrap">{pt.analysis?.fraudReason || 'No anomalies detected.'}</p>
                            </div>
                        </div>

                        {isLast && (
                          <div className="hidden lg:flex w-72 bg-white p-6 shrink-0 flex-col overflow-y-auto rounded-lg border-2 border-white shadow-xl">
                              <h3 className="font-bold uppercase text-xs mb-2 truncate text-gray-500 border-b pb-2">{pt.file?.name} - Reason</h3>
                              <p className="text-sm whitespace-pre-wrap leading-relaxed">{pt.analysis?.fraudReason || 'No anomalies detected.'}</p>
                          </div>
                        )}
                        
                        {!isLast && receipts.length === 1 && (
                          <div className="hidden lg:flex w-72 bg-white p-6 shrink-0 flex-col overflow-y-auto rounded-lg border-2 border-white shadow-xl opacity-0 pointer-events-none"></div>
                        )}
                      </Fragment>
                    );
                  });
                })()}
            </div>
        </div>
      )}
    </div>
  );
}

