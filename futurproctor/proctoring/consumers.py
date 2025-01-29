from channels.generic.websocket import AsyncWebsocketConsumer
import json

class ExamDetectionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_group_name = 'exam_detections'

        # Join the exam detection group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        # Leave the exam detection group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    # Receive message from WebSocket (detect message to update frontend)
    async def receive(self, text_data):
        data = json.loads(text_data)
        message = data['message']
        
        # Send message to WebSocket (e.g., to update the frontend with new detection info)
        await self.send(text_data=json.dumps({
            'message': message
        }))

    # Function to send detection message to WebSocket
    async def send_detection_update(self, message):
        await self.send(text_data=json.dumps({
            'message': message
        }))
