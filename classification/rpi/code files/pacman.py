from RPLCD.i2c import CharLCD
from time import sleep

lcd = CharLCD('PCF8574', 0x27)
lcd.clear()

# Custom characters (5x8)
pacman_open = [
    0b01110,
    0b11111,
    0b11000,
    0b10000,
    0b11000,
    0b11111,
    0b01110,
    0b00000
]

pacman_closed = [
    0b01110,
    0b11111,
    0b11111,
    0b11111,
    0b11111,
    0b11111,
    0b01110,
    0b00000
]

water_bottle = [
    0b00100,
    0b01110,
    0b01110,
    0b01110,
    0b01110,
    0b01110,
    0b01110,
    0b00000
]

banana = [
    0b00100,  # tip
    0b01100,  # upper curve
    0b01100,  # thick body
    0b01100,  # thick body
    0b01110,  # wider midsection
    0b00110,  # taper bottom
    0b00010,  # small tail
    0b00000
]

paperball = [
    0b00000,
    0b01110,
    0b10101,
    0b01010,
    0b10101,
    0b01110,
    0b00000,
    0b00000
]

pen = [
    0b00000,  # small top cap
    0b01100,  # upper body
    0b01100,  # body
    0b00100,  # body
    0b00100,  # body
    0b00100,  # nib area
    0b00100,  # sharp tip
    0b00000
]


dustbin = [
    0b00000,  # curved lid
    0b11111,  # lid rim
    0b11011,  # slanted holes start
    0b10101,  # alternating pattern
    0b11011,  # vertical detail
    0b10101,  # narrow curved base
    0b01110,
    0b00000
]

# Load custom characters
lcd.create_char(0, pacman_open)
lcd.create_char(1, pacman_closed)
lcd.create_char(2, water_bottle)
lcd.create_char(3, banana)
lcd.create_char(4, paperball)
lcd.create_char(5, pen)
lcd.create_char(6, dustbin)

items = [2, 3, 4, 5, 6]  # food sequence

try:
    while True:
        lcd.clear()
        # Draw items initially
        for i, it in enumerate(items):
            lcd.write_string(chr(it))
            lcd.write_string(" ")

        for pos in range(0, 16):
            lcd.clear()

            # Draw items ahead of Pac-Man
            for i, it in enumerate(items):
                food_pos = 3 + (i * 3)
                if food_pos > pos:
                    lcd.cursor_pos = (0, food_pos)
                    lcd.write_string(chr(it))

            # Alternate mouth
            lcd.cursor_pos = (0, pos)
            lcd.write_string(chr(0 if pos % 2 == 0 else 1))

            # Slow down movement
            sleep(0.35)

        sleep(0.7)  # small pause before reset (feels smoother)

except KeyboardInterrupt:
    lcd.clear()
    lcd.write_string("Kuppa Thotti")
