
class CUASession:
    def __init__(self, session_id: str):
        self.session_id = session_id
    
    def __repr__(self):
        return f"CUASession(session_id='{self.session_id}')"