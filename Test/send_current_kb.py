import os
import asyncio
from bot.whatsapp import WhatsAppClient
from google.cloud import firestore
from dotenv import load_dotenv

load_dotenv()

async def seed_and_send():
    admin_phone = os.environ.get("ADMIN_PHONE_NUMBER")
    if not admin_phone:
        print("ADMIN_PHONE_NUMBER not set.")
        return

    wa = WhatsAppClient()
    
    # 1. Read the original KB from local path
    original_kb_path = "/Users/bhumitgoyal/Downloads/GoHappyClub_KnowledgeBase.md"
    try:
        with open(original_kb_path, "r") as f:
            kb_content = f.read()
    except FileNotFoundError:
        print(f"Original KB not found at {original_kb_path}")
        return

    # 2. Seed it into Firestore so get_master_kb() works correctly moving forward
    print("Seeding Firestore with original Knowledge Base...")
    db = firestore.AsyncClient(project=os.environ.get("GCP_PROJECT_ID"))
    doc_ref = db.collection("system_data").document("knowledge_base")
    await doc_ref.set({"content": kb_content})
    print("Firestore seeded successfully!")

    # 3. Send the file to the admin
    print(f"Uploading {original_kb_path} to WhatsApp...")
    media_id = await wa.upload_media(original_kb_path, mime_type="text/plain")
    
    if media_id:
        print(f"Media uploaded. ID: {media_id}. Sending document...")
        success = await wa.send_document(
            to=admin_phone.replace("+", ""),
            media_id=media_id,
            filename="GoHappyClub_KnowledgeBase.md",
            caption="📄 Here is the original Knowledge Base file that Vertex AI RAG is currently using. It has now been synced as the Master Copy in the database."
        )
        if success:
            print("Document sent successfully!")
        else:
            print("Failed to send document.")
    else:
        print("Failed to upload media.")
        
    await wa.close()

if __name__ == "__main__":
    asyncio.run(seed_and_send())
