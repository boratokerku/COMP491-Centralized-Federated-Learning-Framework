// Prints a bcrypt hash for an internal-area account (see deploy/.env.example).
//   docker run --rm -it -v "$PWD/frontend":/app -w /app node:22-alpine node scripts/hash-password.mjs
import bcrypt from "bcryptjs";
import readline from "node:readline";

const rl = readline.createInterface({ input: process.stdin, output: process.stderr });
rl.question("Password: ", (password) => {
  rl.close();
  if (password.length < 12) {
    console.error("Use at least 12 characters.");
    process.exit(1);
  }
  console.log(bcrypt.hashSync(password, 12));
});
