const HTTP = require('http')
const PS = require('child_process')
const FS = require('fs')

const helpText = `
usage:
    node wnn <command>

commands:
    start   -   Starts the WNN server
    stop    -   Stops the WNN server

    new <type> <name>   -   Start interactive model creation for <type> and save as <name>
    reset <type> <name>   -   Resets model back to initialized stated

    set <type> <model> live  -   Adds model to models.conf (Reboots WNN server if started)
    set <type> <model> off   -   Remove model from models.conf (Reboots WNN server if started)
`
const sendHelp = () => console.log(helpText)

const startServer = (andThen, orElse) => {
    PS.execFile('npm', ['run serve'])
}
    
const stopServer = (andThen, orElse) => {
    HTTP.get('localhost:8888/stop', (response) => {
        if (response.statusCode === 200 && andThen) andThen()
        else if (orElse) orElse(response)
        else throw Error('ERROR!' + response.statusCode)
    })
}

const ifOnline = (andThen, orElse) => {
    HTTP.get('localhost:8888/online', (response) => {
        if (response.statusCode === 200 && andThen) andThen()
        else if (orElse) orElse(response)
        else throw Error('ERROR!' + response.statusCode)
    })
}

const reboot = () => {
    HTTP.get('localhost:8888/reboot');
}

const setLive = (path) => {
    if (FS.existsSync('./models.conf')) {
        const models = FS.readFileSync('./models.conf', 'utf8').split('\n')
        for (let l in models) {
            if (models[l] === path) return
        }
        FS.appendFileSync('./models.conf', path + '\n', 'utf8')
    }
}
const setOff = (path) => {
    if (FS.existsSync('./models.conf')) {
        const models = FS.readFileSync('./models.conf', 'utf8').split('\n')
        for (let l in models) {
            if (models[l] === path) {
                models.splice(l, 1)
                break
            }
        }
        FS.writeFileSync('./models.conf', models.join('\n'), 'utf8')
    }
}

const command = process.argv[2]
switch (command) {

    case 'start': {
        // reboot or start the WNN server
        ifOnline(reboot, startServer)
        break
    }

    case 'stop': {
        // shut down the server
        ifOnline(stopServer)
        break
    }

    case 'new': {
        const type = process.argv[3]
        const name = process.argv[4]

        if (type && name) {
            // TODO: start interactive model creation 
           throw Error('Command not found!')
        } else
            sendHelp()
        break
    }

    case 'set': {
        const type = process.argv[3]
        const name = process.argv[4]
        const state = process.argv[5]

        if (type && name && state) {
            const path = `${type}/version/${name}`
            if (state === 'live') {
                setLive(path)
                ifOnline(reboot)
            } else if (state === 'off') {
                setOff(path)
                ifOnline(reboot)
            } else
                sendHelp()
        } else {
            sendHelp()
        }
        break
    }

    case 'reset': {
        const type = process.argv[3]
        const name = process.argv[4]

        if (type && name) {
            const path = `${type}/version/${name}`
            PS.execFile('rm', [`${path}/weights`, `${path}/log`])
            ifOnline(reboot)
        } else {
            sendHelp()
        }
        break
    }

    default: 
        sendHelp()
}