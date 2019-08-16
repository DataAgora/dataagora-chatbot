export class Conversation {
    constructor(prior = null, model) {
        if (prior == null) {
            prior = "\
            You said: Hello!\n\
            I said: Hello!\n\
            "
        }

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
    
    // if (answer == "") {
    //     answer = suggested_answer;
    // } else {
    //     answer = party + answer;
    // }
      
      
    // if (party != answer[0]) {
    //     prefix = next_party + ": ";
    //     answer = prefix + answer;
    // }
      
    // answer = answer.trim()
    
    // this.conversation.push((party, party + answer))    
    // await this.get_suggestion()
    // return answer;
  }
    
    
  async retry() {
    await this.get_suggestion()
  }
    
  isPunctuation(char) {
    return char == '.' || char == '?' || char == '!';
  }
  async get_answer(conv) {
    var answer = await this.model.forwardPass([conv])
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
        console.log("old", line);
        while (!this.isPunctuation(line[j])) {
            j--;
        }
        line = line.slice(0, j + 1);
        console.log("new", line);
        return line;
    }
    
    return "";
  }
}
  
    