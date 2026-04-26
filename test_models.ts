import { GoogleGenAI } from "@google/genai";
import * as dotenv from 'dotenv';
dotenv.config();
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
async function test() {
  try {
    const res = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: "hello"
    });
    console.log("2.5 flash works");
  } catch(e: any) { 
    console.error("2.5 flash fail: ", e.message); 
  }
  
  try {
    const res = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: "hello"
    });
    console.log("3 cash preview works");
  } catch(e: any) { 
    console.error("3 flash preview fail: ", e.message); 
  }

  try {
    const embedRes = await ai.models.embedContent({
      model: 'text-embedding-004',
      contents: "hello"
    });
    console.log("text embedding works");
  } catch(e: any) { console.error("embed 004 fail: ", e.message); }

  try {
    const embedRes = await ai.models.embedContent({
      model: 'gemini-embedding-2-preview',
      contents: "hello"
    });
    console.log("gemini embed works");
  } catch(e: any) { console.error("gemini embed fail: ", e.message); }
}
test();
