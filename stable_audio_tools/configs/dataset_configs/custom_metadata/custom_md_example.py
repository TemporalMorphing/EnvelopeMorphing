def get_custom_metadata(info, audio):
    if audio is None:
        print("Warning: Audio is None in custom metadata function")
        return None
    
    try:
        return {"prompt": "generate audio"}
    except Exception as e:
        print(f"Error in custom metadata function: {str(e)}")
        return None