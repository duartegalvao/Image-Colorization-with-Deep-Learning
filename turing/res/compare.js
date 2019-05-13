const NUM_SAMPLES = 120;
const SAVE_FILENAME = "tests.json";

const TT_STATE = {
    IDLE: 1,
    GENERATING: 2,
    TESTING: 3
};

const TT_CHOICE = {
    LEFT: 1,
    RIGHT: 2
};


class TuringTest {
    constructor(num_samples) {
        this.num_samples = num_samples;
        this.tests = [];
        this.state = TT_STATE.IDLE;
        this.current_test = {
            start_time: -1,
            img_id: -1,
            correct_choice: -1
        };
    }

    genTest() {
        function get_img_src(img_id, is_ground_truth) {
            let src = "/src/images/test_" + img_id + "/";
            if (is_ground_truth)
                src += "true_color";
            else
                src += "after_train"
            return src + ".png";
        }

        this.state = TT_STATE.GENERATING;

        let img_id = Math.floor(Math.random() * this.num_samples);
        let correct_choice = Math.floor(Math.random() * 2) + 1;

        switch (correct_choice) {
            case TT_CHOICE.LEFT:
                $("#img_left").attr('src', get_img_src(img_id, true));
                $("#img_right").attr('src', get_img_src(img_id, false));
                break;
            case TT_CHOICE.RIGHT:
                $("#img_left").attr('src', get_img_src(img_id, false));
                $("#img_right").attr('src', get_img_src(img_id, true));
                break;
            default:
                console.log("Unknown error (invalid choice side).")
        }

        this.current_test = {
            start_time: $.now(),
            img_id: img_id,
            correct_choice: correct_choice
        };

        this.state = TT_STATE.TESTING;
    }

    pickChoice(choice) {
        if (this.state === TT_STATE.TESTING) {
            let click_time = $.now();
            this.state = TT_STATE.IDLE;

            this.tests.push({
                img_id: this.current_test.img_id,
                correct_choice: this.current_test.correct_choice,
                correct: this.current_test.correct_choice === choice,
                time: click_time - this.current_test.start_time
            });
        }
    }

    save(filename) {
        function download(content, fileName, contentType) {
            var a = document.createElement("a");
            var file = new Blob([content], {type: contentType});
            a.href = URL.createObjectURL(file);
            a.download = fileName;
            a.click();
        }
        let jsonData = JSON.stringify({tests: this.tests,
            num_samples: this.num_samples,
            generated: $.now()
        });
        download(jsonData, filename, 'text/plain');
    }
}

var turing_test = new TuringTest(NUM_SAMPLES);

$(document).ready(function () {
    $(window).keydown(function(e) {
        let key = e.which;
        if (key === 37 || key === 65) { // left or A
            $("#start-instruction").remove();
            turing_test.pickChoice(TT_CHOICE.LEFT);
            turing_test.genTest();
        }
        else if (key === 39 || key === 68) { // right or D
            $("#start-instruction").remove();
            turing_test.pickChoice(TT_CHOICE.RIGHT);
            turing_test.genTest();
        }
        else if (key === 83) {
            turing_test.save(SAVE_FILENAME);
        }
    });
});
