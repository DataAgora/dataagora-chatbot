/*
CHATBOT
- A higher layer on the language model that actually facilitates conversation
- Essentially the middleman between the user and the language model  
*/
export class Chatbot {
    constructor(prior = null, model) {
        this.model = model
        this.suggestion = null 
        this.me = "You said: ";
        this.you = "I said: ";
        this.parties  = [ this.me, this.you ]
        
        this.conversation = []    
    }
    
    
  
  get_prior() {
    var conv = this.conversation[this.conversation.length - 1][1]
    return conv;
  } 
  
  async get_suggestion() {
    var conv = this.get_prior()
    //conv += this.you;
    var answer = await this.get_answer(conv);
          
    return answer;
  } 
  
  async next(isHuman = false, answer = "") {
    var party;
    if (isHuman) {
        party = this.me;
        console.assert(answer != "");
        
    } else {
        party = this.you;
        answer = await this.get_suggestion();
        
    }
    this.conversation.push([party, party+answer]);
    console.log(this.conversation);
    return answer;
  }
    
  isPunctuation(char) {
    return char == '.' || char == '?' || char == '!';
  }
  async get_answer(conv) {
    var answer = await this.model.forwardPass([conv], 40, 0.7)
    var lines = answer.split('\n');
    var line = "";
    for (var i = 0; i < lines.length; i++) {
        line = lines[i];
        if (line != "") {
            break;
        }
    }
    
    if (line != "") {
        line = line.split(":");
        line = line[line.length - 1];
        line = line.trim();
        var j = line.length - 1;
        while (!this.isPunctuation(line[j]) && j >= 0) {
            j--;
        }
        line = line.slice(0, j + 1);
        return line;
    }
    
    return "";
  }
}
  
    