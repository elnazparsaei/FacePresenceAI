import uuid

def generate_random_nid():
    """
    تولید شناسه ملی تصادفی 10 رقمی (برای ناشناس‌ها)
    """
    return str(uuid.uuid4().int)[:10]
