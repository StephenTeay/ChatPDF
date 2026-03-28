css = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

* {
    box-sizing: border-box;
}

.chat-message {
    padding: 1.25rem 1.5rem;
    border-radius: 0.75rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    animation: fadeSlideIn 0.3s ease forwards;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.chat-message.user {
    background-color: #1e2530;
    border: 1px solid #2e3a4e;
}

.chat-message.bot {
    background-color: #252f3e;
    border: 1px solid #344357;
}

.chat-message .avatar {
    flex-shrink: 0;
    width: 46px;
    height: 46px;
}

.chat-message .avatar img {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid #3b4f6b;
}

.chat-message .message {
    flex: 1;
    color: #dce6f0;
    font-family: 'Inter', sans-serif;
    font-size: 0.93rem;
    line-height: 1.65;
    padding-top: 0.15rem;
}

.chat-message .message strong {
    color: #ffffff;
}

.chat-message .message code {
    background: #1a2233;
    border: 1px solid #2e3d55;
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
    font-size: 0.85rem;
    color: #7dd3fc;
}

.chat-message .message pre {
    background: #1a2233;
    border: 1px solid #2e3d55;
    border-radius: 8px;
    padding: 1rem;
    overflow-x: auto;
}

.source-badge {
    display: inline-block;
    background: #1d3a5f;
    color: #93c5fd;
    border: 1px solid #2563eb44;
    border-radius: 4px;
    font-size: 0.72rem;
    padding: 0.15rem 0.5rem;
    margin-top: 0.5rem;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
}

.source-label {
    color: #64748b;
    font-size: 0.75rem;
    font-family: 'Inter', sans-serif;
    margin-top: 0.5rem;
    display: block;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png"
             alt="AI Assistant"
             style="max-height: 46px; max-width: 46px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=100&h=100&fit=crop&crop=face"
             alt="User"
             style="max-height: 46px; max-width: 46px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''
