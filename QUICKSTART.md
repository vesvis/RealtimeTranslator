# 🚀 Quick-Start Guide: Set Up Your Own API Keys

Get started with Realtime Translator in 10 minutes!

## Step 1: Create Your API Accounts (Free!)

### Deepgram (Speech-to-Text) — $200 FREE credit!
1. Go to [console.deepgram.com/signup](https://console.deepgram.com/signup)
2. Sign up with Google or email
3. Click **"API Keys"** in the sidebar
4. Click **"Create a New API Key"**
5. Name it "Realtime Translator" → **Create Key**
6. **Copy the key** (you won't see it again!)

> 💡 $200 credit = ~775 hours of transcription for free!

---

### OpenAI (Translation) — ~$5 FREE credit
1. Go to [platform.openai.com/signup](https://platform.openai.com/signup)
2. Sign up and verify your account
3. Add a payment method (required for API access, but you get free credits)
4. Go to **API Keys** → **Create new secret key**
5. Name it "Realtime Translator" → **Create**
6. **Copy the key** (you won't see it again!)

> 💡 $5 credit = ~50+ hours of translation!

---

### ElevenLabs (Text-to-Speech) — 10,000 chars FREE/month
1. Go to [elevenlabs.io/sign-up](https://elevenlabs.io/sign-up)
2. Sign up with Google or email
3. Click your **profile icon** (top right) → **Profile + API Key**
4. **Copy the API key**

> 💡 10,000 characters/month = ~30 minutes of TTS free!
> 
> For more usage, consider the **Creator plan ($22/mo)** which gives 100,000 chars/month.

---

### Fish Audio (Optional Alternative TTS)
1. Go to [fish.audio](https://fish.audio)
2. Create an account
3. Go to **Settings** → **API Keys**
4. **Copy the key**

---

## Step 2: Enter Keys in the App

1. **Open Realtime Translator**
2. Scroll down to the **🔑 API Keys (BYOK)** section
3. Paste your keys:
   - **Deepgram**: Your Deepgram key
   - **OpenAI**: Your OpenAI key  
   - **ElevenLabs**: Your ElevenLabs key
   - **Fish Audio**: (optional)
4. ✅ Check **"Remember my keys"** to save them
5. Click **Start Server**!

---

## Cost Breakdown

| Service | Free Tier | After Free Tier |
|---------|-----------|-----------------|
| **Deepgram** | $200 credit (~775 hrs) | $0.0043/min |
| **OpenAI** | ~$5 credit | ~$0.02/hr |
| **ElevenLabs** | 10K chars/mo (~30 min) | $22/mo for 100K chars |

### Estimated Monthly Cost (Heavy Usage: 10 hrs/month)
- With ElevenLabs: ~$5-10/month
- With OpenAI TTS instead: ~$2-3/month

---

## Troubleshooting

### "Missing API keys" error
- Make sure all required keys are entered (Deepgram, OpenAI, and ElevenLabs)
- Check there are no extra spaces in the keys

### Keys not saving
- Make sure **"Remember my keys"** is checked
- Keys are stored locally at: `~/.realtime_translator/config.json`

### Translation not working
- Check the log output for specific error messages
- Verify your OpenAI account has available credits

---

## Need Help?

Contact [your name/contact] if you run into any issues!
