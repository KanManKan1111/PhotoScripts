import Counter
import time
class WebSender:


    request_dictionary = []

    def __init__(self):
        self = self.name

    
    def  blacklistManager(address):
        isAllowed = None
        requests = set(WebSender.request_dictionary)

        if len(WebSender.request_list) < 6:
            WebSender.request_dictionary.append(address)
            isAllowed = True
        else:
            counts = Counter(requests)
            duplicates = [item for item, count in counts.items() if count > 1]
            if duplicates > 2:
                isAllowed = False

        return isAllowed

    
    def sendRequests(addresses):
        response = []
        start = time.time

        for address in addresses:
            if not isinstance(address,str):
                next
            if WebSender.blacklistManager(address) is True:
                response.append("Valid 200")
            elif WebSender.blacklistManager(address) is False:
                response.append("Invalid 403")
        
        end = time.time

        return response
            


            




def main():
    addresses = [7, "www.abc.com", "www.xyz.com", "www.abd.com", "www.cctv.com", "www.abc.com", "www.hello.com", "www.pen15.com"]
