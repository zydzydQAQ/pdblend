"""Lossless aggregation for clients requesting a non-streaming response."""
class CompletionAccumulator:
    def __init__(self):
        self.request_id=None
        self.text=[]
        self.token_ids=[]
        self.token_received_s=[]
        self.usage=None
        self.finish_reason=None

    def add(self,event,received_s):
        if event.get('error'):
            raise RuntimeError(str(event['error']))
        self.request_id=event.get('id',self.request_id)
        ids=event.get('token_ids') or []
        self.token_ids.extend(ids)
        self.token_received_s.extend([received_s]*len(ids))
        for choice in event.get('choices',[]):
            if choice.get('index',0)!=0:
                raise ValueError('only one completion per request is supported')
            self.text.append(choice.get('text',''))
            self.finish_reason=choice.get('finish_reason') or self.finish_reason
        if event.get('usage'):
            self.usage=event['usage']

    def result(self):
        if self.usage is None or self.usage.get('completion_tokens')!=len(self.token_ids):
            raise RuntimeError('incomplete downstream token stream')
        return dict(id=self.request_id,choices=[dict(text=''.join(self.text),index=0,
            finish_reason=self.finish_reason)],usage=self.usage,token_ids=self.token_ids,
            token_received_s=self.token_received_s)
