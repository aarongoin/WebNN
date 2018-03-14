// testing weight distribution

// 26 clients -- full alphabet
var clients = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'];
var state = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'];

// 4 clients
// var clients = ['a', 'b', 'c', 'd'];
// var state = ['a', 'b', 'c', 'd'];

var toMerge = '';

function print() {
    var n = 0;
    var a = [];
    var nn = 0;
    console.log("\n")
    for (index in state) {
        n = 0;
        a = [];
        for (char of state[index]) if (a.indexOf(char) === -1 && clients.indexOf(char) !== -1) {
            n++;
            a.push(char);
        }
        if (n === 26) nn++;
        console.log((state[index] || '') + "  unique: " + n + " of: " + state[index].length);
    }
    console.log("\nmix rate: " + nn / clients.length);
    console.log("\n")
}

function train(index) {
    // add more of self and add another
    state[index] += ( toMerge || '' );
    toMerge = clients[index];
}

function shuffle(array) {
    var swap = '';
    var s = 0;
    for (var r = 0; r < 20; r++)
        for (i in array) {
            s = ( Math.random() * array.length ) << 0;
            swap = array[i];
            array[i] = array[s];
            array[s] = swap;
        }

    return array;
}

function trainAll(rounds) {
    for (var r = 1; r <= rounds; r++) {
        var todo = shuffle( Object.assign([], clients) );
        if (Math.random() > 0.15) todo.unshift(todo.splice(todo.indexOf('a'), 1)[0]);
        if (Math.random() > 0.15) todo.unshift(todo.splice(todo.indexOf('b'), 1)[0]);
        if (Math.random() > 0.15) todo.unshift(todo.splice(todo.indexOf('c'), 1)[0]);
        if (Math.random() > 0.15) todo.unshift(todo.splice(todo.indexOf('d'), 1)[0]);
        if (Math.random() > 0.15) todo.unshift(todo.splice(todo.indexOf('e'), 1)[0]);
        if (Math.random() > 0.15) todo.unshift(todo.splice(todo.indexOf('f'), 1)[0]);

        if (Math.random() > 0.15) todo.splice(todo.indexOf('x'));
        if (Math.random() > 0.15) todo.splice(todo.indexOf('y'));
        if (Math.random() > 0.15) todo.splice(todo.indexOf('z'));
        for (client of todo) {
            train(clients.indexOf(client));
        }
    }
    print();
}

if (process.argv[2]) trainAll(process.argv[2]);
else console.error("Missing argument! User must give number of rounds to train.")