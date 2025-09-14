import asyncio
from openai import AsyncOpenAI


async def main():
    client = AsyncOpenAI()

    async with client.realtime.connect(
        model="gpt-4o-realtime-preview-2024-12-17"
    ) as connection:
        # await connection.session.update(session={"type": "realtime", "modalities": ["text"]})

        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Say hello!"}],
            }
        )
        # Fixed: Properly format the response dictionary and string quotes
        # await connection.response.create(response={"voice": [None]})
        await connection.response.create()
        # await connection.response.create(response={"modalities": ["text"]})

        async for event in connection:
            print("EVENT TYPE", event.type, "\n")
            # print(event, "\n")
            if event.type == "response.text.delta":
                print(event.delta, flush=True, end="")
                print()  # Add newline

            # If the server is speaking, you'll get transcript deltas instead:
            elif event.type in (
                "response.output_audio_transcript.delta",
                "response.audio_transcript.delta",
            ):
                print(event.delta, end="", flush=True)
                print()  # Add newline
                print(event)
                print()

            # End-of-text markers for either path:
            elif event.type in (
                "response.text.done",
                "response.output_audio_transcript.done",
                "response.audio_transcript.done",
            ):
                print("end of text marker", event)

            elif event.type == "response.done":
                print()
                break


asyncio.run(main())
