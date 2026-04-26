import { GoogleGenAI } from "@google/genai";
import * as fs from 'fs';
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
async function test() {
  const file = fs.readFileSync('package.json'); // dummy image data conceptually
  try {
    const res = await ai.models.embedContent({
      model: 'text-embedding-004',
      contents: [{inlineData: {data: file.toString('base64'), mimeType: 'image/jpeg'}}]
    });
    console.log("SUCCESS", res.embeddings?.[0]?.values?.length);
  } catch(e: any) {
    console.error("ERROR", e.message);
  }
}
test();
